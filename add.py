"""
LOGOS — add.py
Post-publish indexing. Run after a post goes live on LinkedIn.

Usage:
    python add.py [--title "..."] [--hook "..."] [--cluster A] [--url "https://..."]
    Paste draft, end with /end
"""

import os
import json
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT    = Path(__file__).parent
CONFIG  = json.loads((ROOT / "config" / "local_paths.json").read_text(encoding="utf-8"))
VAULT   = Path(CONFIG["obsidian_vault_root"])
INDEX   = VAULT / CONFIG["vault_paths"]["agent_data_dir"] / "lpl_index.jsonl"
LPL_DIR = VAULT / CONFIG["vault_paths"]["lpl_dir"]
BRAND   = json.loads((ROOT / "brand_context.json").read_text(encoding="utf-8"))


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def gen_id() -> str:
    return f"LPL-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

def first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:120]
    return "(Untitled)"

def suggest_cluster(draft: str) -> str | None:
    try:
        from openai import OpenAI
        clusters = BRAND.get("clusters", {})
        desc = "\n".join(f"  {k}: {v}" for k, v in clusters.items())
        prompt = (
            f"Classify this LinkedIn post into one cluster.\n\nClusters:\n{desc}\n\n"
            f"Post:\n\"\"\"\n{draft[:2000]}\n\"\"\"\n\nReply with exactly one letter: A, B, C, or D."
        )
        resp = OpenAI().chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5, temperature=0,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return answer if answer in {"A", "B", "C", "D"} else None
    except Exception:
        return None


def read_draft() -> str:
    print("Paste your draft. End with /end\n")
    lines = []
    for line in sys.stdin:
        if line.strip() == "/end":
            break
        lines.append(line)
    return "".join(lines).strip()


def write_md(lpl_id: str, title: str, hook: str, cluster: str,
             url: str, draft: str) -> Path:
    yyyy, mm = lpl_id[4:8], lpl_id[8:10]
    path = LPL_DIR / yyyy / mm / f"{lpl_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    rel = str(path.relative_to(VAULT)).replace("\\", "/")
    fm_lines = [
        "---", "type: lpl_post",
        f"lpl_id: {lpl_id}",
        f'date_published: "{now_iso()}"',
        "status: published", "channel: linkedin",
        f'title: "{title}"', f'hook: "{hook}"',
        f"cluster: {cluster}", f"path: {rel}",
    ]
    if url:
        fm_lines += ["links:", f'  url: "{url}"']
    fm_lines.append("---")

    path.write_text("\n".join(fm_lines) + "\n\n" + draft + "\n", encoding="utf-8")
    return path


def append_index(lpl_id: str, title: str, hook: str,
                 cluster: str, path: Path) -> None:
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    rel = str(path.relative_to(VAULT)).replace("\\", "/")
    entry = {
        "lpl_id": lpl_id, "date_published": now_iso(),
        "status": "published", "title": title,
        "hook": hook, "cluster": cluster, "path": rel,
    }
    with INDEX.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_system_status(cluster: str) -> None:
    """Update SYSTEM_STATUS.md snapshot after a new post is added."""
    status_path = ROOT / "SYSTEM_STATUS.md"
    if not status_path.exists():
        return

    # Count posts by cluster
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    if INDEX.exists():
        for line in INDEX.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    p = json.loads(line)
                    c = p.get("cluster", "")
                    if c in counts:
                        counts[c] += 1
                except Exception:
                    continue

    total = sum(counts.values())
    today = datetime.now().strftime("%Y-%m-%d")

    text = status_path.read_text(encoding="utf-8")

    # Update the AUTO-UPDATE block fields
    replacements = {
        "| Posts published |": f"| Posts published | {total} |",
        "| Cluster A": f"| Cluster A (AI widens judgment gap) | {counts['A']} |",
        "| Cluster B": f"| Cluster B (PM defines boundaries) | {counts['B']} |",
        "| Cluster C": f"| Cluster C (Build-in-public) | {counts['C']} |",
        "| Cluster D": f"| Cluster D (AI as infrastructure) | {counts['D']} |",
        "| Last activity |": f"| Last activity | {today} |",
    }

    lines = text.splitlines()
    new_lines = []
    for line in lines:
        replaced = False
        for prefix, new_val in replacements.items():
            if line.strip().startswith(prefix):
                new_lines.append(new_val)
                replaced = True
                break
        if not replaced:
            # Also update the top-level Updated date
            if line.startswith("**Updated:**"):
                new_lines.append(f"**Updated:** {today}  **Status:** Active")
            else:
                new_lines.append(line)

    status_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title",   help="Post title")
    parser.add_argument("--hook",    help="One-liner hook")
    parser.add_argument("--cluster", help="A / B / C / D")
    parser.add_argument("--url",     default="", help="LinkedIn URL")
    parser.add_argument("--id",      help="Override LPL ID")
    args = parser.parse_args()

    draft = read_draft()
    if not draft:
        print("Empty draft. Exiting.")
        return

    interactive = sys.stdin.isatty()

    title = args.title or (input(f"Title [{first_line(draft)}]: ").strip() if interactive else first_line(draft)) or first_line(draft)
    hook  = args.hook  or (input("Hook: ").strip() if interactive else "")
    url   = args.url   or (input("LinkedIn URL (optional): ").strip() if interactive else "")

    if args.cluster:
        cluster = args.cluster.upper()
    else:
        print("Suggesting cluster...", end=" ", flush=True)
        suggestion = suggest_cluster(draft)
        desc = BRAND.get("clusters", {})
        if suggestion:
            print(f"{suggestion}  ({desc.get(suggestion, '')})")
            if interactive:
                confirm = input(f"Cluster [{suggestion}]: ").strip().upper()
                cluster = confirm if confirm in {"A","B","C","D"} else suggestion
            else:
                cluster = suggestion
        else:
            print("(no suggestion)")
            if interactive:
                for k, v in desc.items():
                    print(f"  {k}: {v}")
                cluster = input("Cluster [A/B/C/D]: ").strip().upper()
            else:
                cluster = "D"

    if cluster not in {"A","B","C","D"}:
        print(f"Invalid cluster: {cluster}")
        return

    lpl_id  = args.id or gen_id()
    md_path = write_md(lpl_id, title, hook, cluster, url, draft)
    append_index(lpl_id, title, hook, cluster, md_path)
    update_system_status(cluster)

    print(f"\n{lpl_id}")
    print(f"  .md  -> {md_path}")
    print(f"  idx  -> {INDEX}")
    print(f"  status updated")

    from integrity import run_check
    run_check(fix=True, silent_if_clean=True)


if __name__ == "__main__":
    main()

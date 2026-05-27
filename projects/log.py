"""
LOGOS — projects/log.py
Log a project experience summary → find matching concepts → write to concept pages.

This feeds the flywheel:
  side project / day-job milestone
      -> experience.jsonl  (index)
      -> 06_Concepts/*.md  (### My Experience sections)
      -> inspire.py / next.py use this data

Usage:
    python projects/log.py              # interactive mode
    python projects/log.py --list       # show logged experiences
"""

import json
import sys
import re
import argparse
from datetime import date
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

ROOT         = Path(__file__).parent.parent
OBS_CACHE    = ROOT / "kg" / "obsidian_concepts_cache.json"
EXPERIENCE_LOG = ROOT / "projects" / "experience.jsonl"
CONCEPTS_DIR = Path("G:/My Drive/AI Native PM/AI Native PM/06_Concepts")

client = OpenAI()


# ── Utilities ──────────────────────────────────────────────────────────────────

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


def embed(text: str) -> list[float]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return resp.data[0].embedding


# ── Concept loading ────────────────────────────────────────────────────────────

def load_obsidian_concepts() -> list[dict]:
    if not OBS_CACHE.exists():
        print("Obsidian cache not found. Run: python kg/compare.py")
        sys.exit(1)
    return json.loads(OBS_CACHE.read_text(encoding="utf-8"))


def find_concept_file(concept_name: str) -> Path | None:
    """Find the .md file for a given concept name."""
    slug = concept_name.strip().replace(" ", "-")
    # Try direct slug match
    candidate = CONCEPTS_DIR / f"{slug}.md"
    if candidate.exists():
        return candidate
    # Search by frontmatter concept field
    for f in CONCEPTS_DIR.glob("*.md"):
        if f.name.startswith("_"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 2:
                for line in parts[1].splitlines():
                    if line.strip().startswith("concept:"):
                        name = line.split(":", 1)[1].strip()
                        if name.lower() == concept_name.lower():
                            return f
    return None


# ── Concept matching ───────────────────────────────────────────────────────────

def find_top_concepts(summary_emb: list[float],
                      concepts: list[dict],
                      top_n: int = 5) -> list[tuple[float, dict]]:
    scored = [(cosine(summary_emb, c["embedding"]), c) for c in concepts]
    scored.sort(reverse=True)
    return scored[:top_n]


# ── Writing to concept pages ───────────────────────────────────────────────────

def write_experience_to_concept(concept_name: str, project: str,
                                 version_or_milestone: str,
                                 summary: str,
                                 today: str) -> bool:
    """Append a My Experience bullet to the concept's .md file."""
    path = find_concept_file(concept_name)
    if not path:
        print(f"  Could not find file for: {concept_name}")
        return False

    text = path.read_text(encoding="utf-8", errors="ignore")
    bullet = f"- [{project} {version_or_milestone}] {summary} ({today})"

    # Check for existing My Experience section
    if "### My Experience" in text:
        # Append bullet after the section header
        text = text.replace(
            "### My Experience",
            f"### My Experience\n{bullet}",
            1
        )
    else:
        # Add section before ### Related Concepts or at end
        if "### Related Concepts" in text:
            text = text.replace(
                "### Related Concepts",
                f"### My Experience\n{bullet}\n\n### Related Concepts",
                1
            )
        else:
            text = text.rstrip() + f"\n\n### My Experience\n{bullet}\n"

    path.write_text(text, encoding="utf-8")
    return True


# ── Experience log ─────────────────────────────────────────────────────────────

def append_to_log(entry: dict) -> None:
    EXPERIENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIENCE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_log() -> list[dict]:
    if not EXPERIENCE_LOG.exists():
        return []
    entries = []
    for line in EXPERIENCE_LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


# ── Interactive flow ───────────────────────────────────────────────────────────

def interactive_log():
    print("=" * 60)
    print("LOGOS — Project Experience Logger")
    print("=" * 60)
    print()

    # Project info
    project = input("Project name (e.g. MentorFlow, VFB-Price-Model): ").strip()
    if not project:
        print("Cancelled.")
        return

    ver_label = input("Version or milestone (e.g. v0.91, Samsung-pitch): ").strip()
    if not ver_label:
        ver_label = "milestone"

    print()
    print("Describe what you built / discovered / learned.")
    print("Focus on the insight, not just what you did. (End with blank line)")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    summary = " ".join(lines).strip()

    if not summary:
        print("Cancelled — no summary entered.")
        return

    # Embed and match
    print()
    print("Finding matching concepts...")
    concepts    = load_obsidian_concepts()
    summary_emb = embed(summary)
    matches     = find_top_concepts(summary_emb, concepts, top_n=6)

    print()
    print("Top matching concepts:")
    for i, (score, c) in enumerate(matches, 1):
        sources = c.get("sources", 1)
        print(f"  {i}.  {score:.2f}  {c['concept']}  [{sources} market sources]")
        desc = c.get("description", "")
        if desc:
            print(f"       {desc[:80]}")

    print()
    choice = input("Add to which? (e.g. 1,2  or  all  or  skip): ").strip().lower()

    if choice == "skip" or choice == "":
        print("Skipped — nothing written.")
        selected_concepts = []
    elif choice == "all":
        selected_concepts = [c["concept"] for _, c in matches]
    else:
        idxs = [int(x.strip()) - 1 for x in choice.split(",") if x.strip().isdigit()]
        selected_concepts = [matches[i][1]["concept"] for i in idxs if 0 <= i < len(matches)]

    # Write to concept pages
    today = str(date.today())
    written = []
    for concept_name in selected_concepts:
        ok = write_experience_to_concept(
            concept_name, project, ver_label, summary, today
        )
        if ok:
            print(f"  Written to {concept_name}.md  ok")
            written.append(concept_name)

    # Append to experience log
    if written or selected_concepts:
        entry = {
            "date":            today,
            "project":         project,
            "version":         ver_label,
            "summary":         summary,
            "concept_matches": written,
            "embedding":       summary_emb,
        }
        append_to_log(entry)
        print()
        print(f"Logged to experience.jsonl  ({today})")

    print()
    print("Done. Run 'python projects/next.py' to see what to build next.")


def list_log():
    entries = load_log()
    if not entries:
        print("No experiences logged yet.")
        return
    print(f"{len(entries)} logged experience(s):\n")
    for e in entries:
        ver  = e.get("version", "")
        tag  = f"  [{e['project']} {ver}]  {e['date']}"
        print(tag)
        print(f"  {e['summary'][:90]}")
        concepts = e.get("concept_matches", [])
        if concepts:
            print(f"  -> {', '.join(concepts)}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="Show all logged experiences")
    args = parser.parse_args()

    if args.list:
        list_log()
    else:
        interactive_log()


if __name__ == "__main__":
    main()

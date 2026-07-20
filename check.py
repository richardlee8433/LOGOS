"""
LOGOS — check.py
Pre-publish brand validation.

Usage:
    python check.py
    Paste draft, end with /end
"""

import os
import json
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from rapidfuzz import fuzz

load_dotenv()

ROOT       = Path(__file__).parent
CONFIG     = json.loads((ROOT / "config" / "local_paths.json").read_text(encoding="utf-8"))
VAULT      = Path(CONFIG["obsidian_vault_root"])
INDEX      = VAULT / CONFIG["vault_paths"]["agent_data_dir"] / "lpl_index.jsonl"
LPL_DIR    = VAULT / CONFIG["vault_paths"]["lpl_dir"]
BRAND      = json.loads((ROOT / "brand_context.json").read_text(encoding="utf-8"))
PROMPT_TPL = (ROOT / "prompts" / "check.txt").read_text(encoding="utf-8")


def read_draft() -> str:
    print("Paste your draft. End with /end\n")
    lines = []
    for line in sys.stdin:
        if line.strip() == "/end":
            break
        lines.append(line)
    return "".join(lines).strip()


def load_posts() -> list[dict]:
    if not INDEX.exists():
        return []
    posts = []
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                posts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return [p for p in posts if p.get("status") == "published"]


def fuzzy_check(draft: str, posts: list[dict]) -> dict | None:
    """Return matching post if similarity >= 85%, else None."""
    for post in posts:
        path = post.get("path")
        if not path:
            continue
        full = VAULT / path
        if not full.exists():
            continue
        text = full.read_text(encoding="utf-8")
        # Strip frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            text = parts[2].strip() if len(parts) >= 3 else text
        score = fuzz.token_set_ratio(draft, text)
        if score >= 85:
            return {"post": post, "score": score}
    return None


def llm_check(draft: str, posts: list[dict]) -> str:
    post_list = "\n".join(
        f"- [{p.get('lpl_id')}] \"{p.get('title')}\" — Hook: {p.get('hook')}"
        for p in posts
    )
    prompt = PROMPT_TPL.format(
        positioning=BRAND["positioning"],
        audience_icp=BRAND["audience"]["icp"],
        audience_problem=BRAND["audience"]["expensive_problem"],
        cluster_A=BRAND["clusters"]["A"],
        cluster_B=BRAND["clusters"]["B"],
        cluster_C=BRAND["clusters"]["C"],
        cluster_D=BRAND["clusters"]["D"],
        post_count=len(posts),
        post_list=post_list,
        draft=draft,
    )
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        max_tokens=1024,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def main():
    draft = read_draft()
    if not draft:
        print("Empty draft. Exiting.")
        return

    posts = load_posts()

    # Fast path: fuzzy duplicate check (no API call)
    match = fuzzy_check(draft, posts)
    if match:
        p = match["post"]
        print("\nDUPLICATE DETECTED (no API call)")
        print(f"Similarity : {match['score']:.0f}%")
        print(f"Matched    : {p.get('lpl_id')} — {p.get('title')}")
        print(f"Hook       : {p.get('hook')}")
        print("\nDecision: DUPLICATE")
        return

    # LLM brand check
    print("\nChecking against brand positioning...\n")
    print(llm_check(draft, posts))


if __name__ == "__main__":
    main()

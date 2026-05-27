"""
LOGOS — kg/build.py
Extract concepts from all LPL posts and build a local knowledge graph (logos_kg.db).

Each post → 3-5 key concepts → embeddings → SQLite

Usage:
    python kg/build.py            # process all posts, skip already-processed
    python kg/build.py --rebuild  # wipe and rebuild from scratch
"""

import json
import sqlite3
import argparse
import time
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

ROOT    = Path(__file__).parent.parent
CONFIG  = json.loads((ROOT / "config" / "local_paths.json").read_text(encoding="utf-8"))
VAULT   = Path(CONFIG["obsidian_vault_root"])
INDEX   = VAULT / CONFIG["vault_paths"]["agent_data_dir"] / "lpl_index.jsonl"
DB_PATH = ROOT / "kg" / "logos_kg.db"

client = OpenAI()


# ── Database ──────────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            concept     TEXT NOT NULL,
            description TEXT,
            post_id     TEXT NOT NULL,
            post_hook   TEXT,
            cluster     TEXT,
            embedding   TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_post_id ON concepts(post_id)")
    conn.commit()


def already_processed(conn: sqlite3.Connection, post_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM concepts WHERE post_id = ? LIMIT 1", (post_id,)
    ).fetchone()
    return row is not None


def insert_concepts(conn: sqlite3.Connection, post_id: str, hook: str,
                    cluster: str, concepts: list[dict]) -> None:
    for c in concepts:
        conn.execute(
            """INSERT INTO concepts (concept, description, post_id, post_hook, cluster, embedding)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (c["concept"], c["description"], post_id, hook, cluster,
             json.dumps(c["embedding"]))
        )
    conn.commit()


# ── LLM calls ─────────────────────────────────────────────────────────────────

def extract_concepts(post_text: str, hook: str) -> list[dict]:
    """Ask GPT to extract 3-5 key concepts from a post."""
    prompt = f"""You are analyzing a LinkedIn post by an AI-native Product Manager.

Post hook: {hook}

Post:
\"\"\"
{post_text[:3000]}
\"\"\"

Extract 3-5 distinct key concepts from this post. Each concept should be:
- A specific idea, framework, or claim (not a generic topic)
- Something another PM could search for or reference
- 2-6 words maximum

Respond with a JSON array only, no other text:
[
  {{"concept": "short concept name", "description": "one sentence explaining this concept as used in the post"}},
  ...
]"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content.strip()
    data = json.loads(raw)
    # Handle both {"concepts": [...]} and [...] responses
    if isinstance(data, dict):
        items = data.get("concepts", data.get("items", list(data.values())[0]))
    else:
        items = data
    return items[:5]


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts."""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [r.embedding for r in resp.data]


# ── Post loading ──────────────────────────────────────────────────────────────

def load_posts() -> list[dict]:
    posts = []
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                posts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return [p for p in posts if p.get("status") == "published"]


def find_post_file(post: dict) -> Path | None:
    """Find the .md file for a post — try index path first, then reconstruct from lpl_id."""
    # Try index path
    path = post.get("path")
    if path:
        full = VAULT / path
        if full.exists():
            return full

    # Reconstruct from lpl_id: LPL-YYYYMMDDThhmmssZ[-001] → YYYY/MM/
    lpl_id = post.get("lpl_id", "")
    if len(lpl_id) >= 12:
        yyyy = lpl_id[4:8]
        mm   = lpl_id[8:10]
        lpl_dir = CONFIG["vault_paths"]["lpl_dir"]
        candidate = VAULT / lpl_dir / yyyy / mm / f"{lpl_id}.md"
        if candidate.exists():
            return candidate
        # Some files have a longer name — glob for lpl_id prefix
        pattern = f"{lpl_id}*.md"
        matches = list((VAULT / lpl_dir / yyyy / mm).glob(pattern))
        if matches:
            return matches[0]

    return None


def read_post_text(post: dict) -> str | None:
    f = find_post_file(post)
    if not f:
        return None
    text = f.read_text(encoding="utf-8")
    # Strip frontmatter
    if text.startswith("---"):
        parts = text.split("---", 2)
        text = parts[2].strip() if len(parts) >= 3 else text
    return text


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Wipe DB and rebuild from scratch")
    args = parser.parse_args()

    if args.rebuild and DB_PATH.exists():
        DB_PATH.unlink()
        print("Wiped existing DB.")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    posts = load_posts()
    print(f"Found {len(posts)} published posts\n")

    skipped = processed = failed = 0

    for post in posts:
        post_id = post.get("lpl_id", "?")
        hook    = post.get("hook", "")
        cluster = post.get("cluster", "?")

        if already_processed(conn, post_id):
            skipped += 1
            continue

        text = read_post_text(post)
        if not text:
            print(f"  SKIP {post_id} — file not found")
            skipped += 1
            continue

        print(f"  Processing {post_id} [{cluster}] {hook[:50]}...")

        try:
            concepts = extract_concepts(text, hook)
            if not concepts:
                raise ValueError("No concepts returned")

            # Embed all concepts in one batch call
            embeddings = embed([c["concept"] + ". " + c["description"]
                                for c in concepts])
            for c, emb in zip(concepts, embeddings):
                c["embedding"] = emb

            insert_concepts(conn, post_id, hook, cluster, concepts)
            print(f"    → {len(concepts)} concepts: {[c['concept'] for c in concepts]}")
            processed += 1
            time.sleep(0.3)  # gentle rate limiting

        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

    conn.close()

    # Summary
    print(f"\nDone. Processed: {processed} | Skipped: {skipped} | Failed: {failed}")
    if DB_PATH.exists():
        conn2 = sqlite3.connect(DB_PATH)
        total = conn2.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        conn2.close()
        print(f"Total concepts in KG: {total}")


if __name__ == "__main__":
    main()

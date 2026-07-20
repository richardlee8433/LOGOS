"""
LOGOS — kg/import_pending.py
Import pre-extracted concepts (kg/pending_concepts.json) into logos_kg.db.

The concept-extraction step of build.py was done offline (Claude, 2026-07-20)
while OpenAI quota was exhausted. This script only needs the embedding API.
Safe to re-run: posts already in the DB are skipped.

Usage:
    python kg/import_pending.py
"""

import json
import sqlite3
import sys
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

ROOT    = Path(__file__).parent.parent
CONFIG  = json.loads((ROOT / "config" / "local_paths.json").read_text(encoding="utf-8"))
VAULT   = Path(CONFIG["obsidian_vault_root"])
INDEX   = VAULT / CONFIG["vault_paths"]["agent_data_dir"] / "lpl_index.jsonl"
DB_PATH = ROOT / "kg" / "logos_kg.db"
PENDING = ROOT / "kg" / "pending_concepts.json"

client = OpenAI()


def load_index() -> dict[str, dict]:
    index = {}
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                p = json.loads(line)
                index[p.get("lpl_id")] = p
            except json.JSONDecodeError:
                continue
    return index


def main():
    if not PENDING.exists():
        print("No pending_concepts.json found. Nothing to import.")
        return

    pending = json.loads(PENDING.read_text(encoding="utf-8"))
    index = load_index()
    conn = sqlite3.connect(DB_PATH)

    done = skipped = 0
    for entry in pending:
        post_id = entry["post_id"]
        if conn.execute("SELECT 1 FROM concepts WHERE post_id=? LIMIT 1",
                        (post_id,)).fetchone():
            skipped += 1
            continue

        meta = index.get(post_id)
        if not meta:
            print(f"  SKIP {post_id} — not in index")
            skipped += 1
            continue

        concepts = entry["concepts"][:5]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[c["concept"] + ". " + c["description"] for c in concepts],
        )
        for c, r in zip(concepts, resp.data):
            conn.execute(
                """INSERT INTO concepts
                   (concept, description, post_id, post_hook, cluster, embedding)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (c["concept"], c["description"], post_id,
                 meta.get("hook", ""), meta.get("cluster", "?"),
                 json.dumps(r.embedding)),
            )
        conn.commit()
        done += 1
        print(f"  {post_id} [{meta.get('cluster', '?')}] -> {len(concepts)} concepts")
        time.sleep(0.3)

    total = conn.execute("SELECT COUNT(DISTINCT post_id) FROM concepts").fetchone()[0]
    conn.close()
    print(f"\nImported: {done} | Skipped: {skipped}")
    print(f"Distinct posts in KG: {total}")


if __name__ == "__main__":
    main()

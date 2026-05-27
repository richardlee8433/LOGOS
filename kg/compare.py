"""
LOGOS — kg/compare.py
Compare LOGOS concept KG against a market signal source.

Sources:
  --source obsidian   Use G:/My Drive/AI Native PM/06_Concepts/ (default)
  --source praxis     Use PRAXIS halos.db events

Usage:
    python kg/compare.py
    python kg/compare.py --source praxis
    python kg/compare.py --threshold 0.42
"""

import json
import sqlite3
import argparse
import sys
import re
import time
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

ROOT        = Path(__file__).parent.parent
LOGOS_DB    = ROOT / "kg" / "logos_kg.db"
PRAXIS_DB   = Path("G:/My Drive/PRAXIS/backend/halos.db")
CONCEPTS_DIR = Path("G:/My Drive/AI Native PM/AI Native PM/06_Concepts")
OBSIDIAN_CACHE = ROOT / "kg" / "obsidian_concepts_cache.json"

client = OpenAI()


# ── Utilities ─────────────────────────────────────────────────────────────────

def cosine(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [r.embedding for r in resp.data]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_logos(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT concept, description, cluster, post_id, embedding FROM concepts"
    ).fetchall()
    return [
        {"concept": r[0], "description": r[1], "cluster": r[2],
         "post_id": r[3], "embedding": json.loads(r[4])}
        for r in rows
    ]


def parse_concept_md(path: Path) -> dict | None:
    """Parse an Obsidian concept .md file into a dict."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    # Extract frontmatter
    fm = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    fm[k.strip()] = v.strip()
            text = parts[2]

    concept = fm.get("concept", path.stem.replace("-", " "))
    sources = int(fm.get("sources", 1))

    # Extract "What it is"
    what_match = re.search(r"### What it is\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL)
    what = what_match.group(1).strip() if what_match else ""

    # Extract Key Insights (strip wiki links)
    insights_match = re.search(r"### Key Insights\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL)
    if insights_match:
        raw_insights = insights_match.group(1)
        # Remove [[...]] wiki links
        clean = re.sub(r"\[\[.*?\]\]", "", raw_insights)
        # Keep bullet text only
        insights = " ".join(
            line.lstrip("- ").strip()
            for line in clean.splitlines()
            if line.strip().startswith("-")
        )
    else:
        insights = ""

    embed_text = f"{concept}. {what} {insights}".strip()
    if len(embed_text) < 20:
        return None

    return {
        "concept": concept,
        "description": what[:150] if what else concept,
        "sources": sources,
        "embed_text": embed_text[:1500],
        "embedding": None,
    }


def load_obsidian_concepts(force_rebuild: bool = False) -> list[dict]:
    """Load and embed 06_Concepts, using cache to avoid re-embedding."""
    if OBSIDIAN_CACHE.exists() and not force_rebuild:
        print("Loading Obsidian concepts from cache...")
        cached = json.loads(OBSIDIAN_CACHE.read_text(encoding="utf-8"))
        return cached

    print(f"Reading {CONCEPTS_DIR}...")
    md_files = [f for f in CONCEPTS_DIR.glob("*.md") if f.name != "_index.md"]
    concepts = []
    for f in md_files:
        c = parse_concept_md(f)
        if c:
            concepts.append(c)

    print(f"Embedding {len(concepts)} concepts...")
    texts = [c["embed_text"] for c in concepts]

    # Batch embed (max 100 per call)
    all_embeddings = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i+50]
        all_embeddings.extend(embed_batch(batch))
        time.sleep(0.3)

    for c, emb in zip(concepts, all_embeddings):
        c["embedding"] = emb
        del c["embed_text"]  # don't cache full text

    OBSIDIAN_CACHE.write_text(
        json.dumps(concepts, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Cached {len(concepts)} concepts to {OBSIDIAN_CACHE.name}\n")
    return concepts


def load_praxis_concepts() -> list[dict]:
    if not PRAXIS_DB.exists():
        print(f"PRAXIS DB not found at {PRAXIS_DB}")
        return []
    conn = sqlite3.connect(PRAXIS_DB)
    rows = conn.execute(
        "SELECT label, description, embedding FROM events WHERE embedding IS NOT NULL"
    ).fetchall()
    conn.close()
    return [
        {"concept": r[0], "description": r[1],
         "sources": 1, "embedding": json.loads(r[2])}
        for r in rows
    ]


# ── Analysis ──────────────────────────────────────────────────────────────────

def best_match(query_emb: list, candidates: list[dict]) -> tuple[float, dict]:
    best_score, best_item = 0.0, candidates[0]
    for c in candidates:
        s = cosine(query_emb, c["embedding"])
        if s > best_score:
            best_score, best_item = s, c
    return best_score, best_item


def aggregate_logos(logos: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for item in logos:
        key = item["concept"].lower()
        if key not in seen:
            seen[key] = {**item, "count": 1, "clusters": {item["cluster"]}}
        else:
            seen[key]["count"] += 1
            seen[key]["clusters"].add(item["cluster"])
    return sorted(seen.values(), key=lambda x: x["count"], reverse=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["obsidian", "praxis"],
                        default="obsidian",
                        help="Market signal source (default: obsidian)")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Re-embed Obsidian concepts even if cache exists")
    args = parser.parse_args()

    if not LOGOS_DB.exists():
        print("LOGOS KG not built yet. Run: python kg/build.py")
        return

    # Load market signals
    if args.source == "obsidian":
        market = load_obsidian_concepts(force_rebuild=args.rebuild_cache)
        source_label = "Obsidian 06_Concepts"
    else:
        market = load_praxis_concepts()
        source_label = "PRAXIS halos.db"

    if not market:
        print("No market concepts loaded.")
        return

    # Load LOGOS KG
    conn = sqlite3.connect(LOGOS_DB)
    logos_raw = load_logos(conn)
    conn.close()
    logos_agg = aggregate_logos(logos_raw)

    print(f"LOGOS concepts  : {len(logos_agg)} unique ({len(logos_raw)} total)")
    print(f"Market source   : {source_label} — {len(market)} concepts")
    print(f"Threshold       : {args.threshold}\n")
    print("=" * 65)

    aligned    = []
    brand_only = []

    for item in logos_agg:
        score, match = best_match(item["embedding"], market)
        if score >= args.threshold:
            aligned.append((score, item, match))
        else:
            brand_only.append((score, item, match))

    # Gaps: market concepts with no close LOGOS match
    gaps = []
    for m_item in market:
        score, _ = best_match(m_item["embedding"], logos_raw)
        if score < args.threshold:
            gaps.append((score, m_item))

    # Dedupe gaps
    seen_gaps: set[str] = set()
    unique_gaps = []
    for score, item in sorted(gaps, key=lambda x: x[1].get("sources", 1), reverse=True):
        key = item["concept"].lower()
        if key not in seen_gaps:
            seen_gaps.add(key)
            unique_gaps.append((score, item))

    # ── Output ────────────────────────────────────────────────────────────────

    print(f"\n{'ALIGNED':=<65}")
    print("Market discusses this AND you write about it\n")
    for score, logos_item, market_match in sorted(aligned, reverse=True):
        clusters = "/".join(sorted(logos_item.get("clusters", set())))
        sources  = market_match.get("sources", 1)
        print(f"  {score:.3f}  [{clusters}] {logos_item['concept']}")
        print(f"         = {market_match['concept']}  (market sources: {sources})")
        print()

    print(f"\n{'GAP — market talks about this, you dont':=<65}")
    print("Sorted by market source count — highest = most important gap\n")
    for score, item in unique_gaps:
        sources = item.get("sources", 1)
        print(f"  [{sources:>2} src]  {item['concept']}")
        print(f"           {item['description'][:90]}")
        print()

    print(f"\n{'BRAND-ONLY — you write about this, market hasnt covered yet':=<65}")
    print("Your differentiated positioning or ahead-of-market ideas\n")
    for score, logos_item, _ in sorted(brand_only, reverse=True)[:15]:
        clusters = "/".join(sorted(logos_item.get("clusters", set())))
        print(f"  {score:.3f}  [{clusters}] {logos_item['concept']}")
        print(f"         {logos_item['description'][:80]}")
        print()

    print("=" * 65)
    print(f"ALIGNED     : {len(aligned):>3}  (writing about what market discusses)")
    print(f"GAP         : {len(unique_gaps):>3}  (market topics missing from your writing)")
    print(f"BRAND-ONLY  : {len(brand_only):>3}  (top 15 shown — your unique angle)")


if __name__ == "__main__":
    main()

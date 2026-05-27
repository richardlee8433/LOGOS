"""
LOGOS — kg/inspire.py
Traverse 06_Concepts relationship graph to find writing path ideas.

Finds concepts the market discusses but you haven't written about (gaps),
then traces paths through Related Concepts links from something you HAVE
written about to that gap. Each path becomes a post skeleton.

Usage:
    python kg/inspire.py                # top 3 ideas with LLM skeletons
    python kg/inspire.py --top 5        # show 5 ideas
    python kg/inspire.py --no-skeleton  # paths only, no LLM call (fast)
    python kg/inspire.py --threshold 0.35
"""

import json
import re
import sqlite3
import argparse
import sys
from pathlib import Path
from collections import deque
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

ROOT         = Path(__file__).parent.parent
LOGOS_DB     = ROOT / "kg" / "logos_kg.db"
OBS_CACHE    = ROOT / "kg" / "obsidian_concepts_cache.json"
CONCEPTS_DIR = Path("G:/My Drive/AI Native PM/AI Native PM/06_Concepts")

client = OpenAI()


# ── Utilities ──────────────────────────────────────────────────────────────────

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


# ── Concept graph from .md files ───────────────────────────────────────────────

def parse_all_concepts() -> dict[str, dict]:
    """Parse all concept .md files.
    Returns: { canonical_name: { slug, related: [names], insights: [str] } }
    """
    # Pass 1: slug → canonical name + raw text
    slug_to_name: dict[str, str] = {}
    file_data: dict[str, tuple[Path, str]] = {}

    for f in CONCEPTS_DIR.glob("*.md"):
        if f.name.startswith("_"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        slug = f.stem
        name = slug.replace("-", " ")   # fallback
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 2:
                for line in parts[1].splitlines():
                    if line.strip().startswith("concept:"):
                        name = line.split(":", 1)[1].strip()
                        break
        slug_to_name[slug] = name
        file_data[slug]    = (f, text)

    # Pass 2: extract related concepts + key insights
    concepts: dict[str, dict] = {}
    for slug, (f, text) in file_data.items():
        name = slug_to_name[slug]

        # Related Concepts
        related: list[str] = []
        rel_m = re.search(r"### Related Concepts\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL)
        if rel_m:
            for link in re.findall(r"\[\[([^\]|#]+)(?:\|[^\]]+)?\]\]", rel_m.group(1)):
                link = link.strip()
                related.append(slug_to_name.get(link, link.replace("-", " ")))

        # Key Insights (strip wiki links, dedupe)
        insights: list[str] = []
        ins_m = re.search(r"### Key Insights\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL)
        if ins_m:
            clean = re.sub(r"\[\[.*?\]\]", "", ins_m.group(1))
            seen: set[str] = set()
            for line in clean.splitlines():
                if line.strip().startswith("-"):
                    b = line.lstrip("- ").strip().rstrip(" —")
                    key = b[:50]
                    if len(b) > 15 and key not in seen:
                        seen.add(key)
                        insights.append(b)
                        if len(insights) >= 4:
                            break

        concepts[name] = {"slug": slug, "related": related, "insights": insights}

    return concepts


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_obsidian_cache() -> list[dict]:
    return json.loads(OBS_CACHE.read_text(encoding="utf-8"))


def load_logos_concepts() -> list[dict]:
    conn = sqlite3.connect(LOGOS_DB)
    rows = conn.execute(
        "SELECT concept, description, cluster, embedding FROM concepts"
    ).fetchall()
    conn.close()
    return [
        {"concept": r[0], "description": r[1],
         "cluster": r[2], "embedding": json.loads(r[3])}
        for r in rows
    ]


# ── Hybrid graph (explicit + embedding fallback) ──────────────────────────────

def enrich_graph(graph: dict[str, list[str]],
                 obsidian: list[dict],
                 min_edges: int = 2,
                 top_k: int = 3,
                 sim_threshold: float = 0.35) -> dict[str, list[str]]:
    """For nodes with fewer than min_edges explicit links, add embedding-based
    nearest neighbours as implicit edges.  Mutates graph in place and returns it.
    """
    emb_by_name = {o["concept"]: o["embedding"] for o in obsidian}
    names       = list(emb_by_name.keys())

    for name in list(graph.keys()):
        if len(graph.get(name, [])) >= min_edges:
            continue
        emb = emb_by_name.get(name)
        if emb is None:
            continue
        scores = [
            (cosine(emb, emb_by_name[n]), n)
            for n in names if n != name and n in emb_by_name
        ]
        scores.sort(reverse=True)
        added = 0
        for score, nb in scores:
            if added >= top_k:
                break
            if score >= sim_threshold and nb not in graph.get(name, []):
                graph.setdefault(name, []).append(nb)
                added += 1

    return graph


# ── Gap / covered classification ───────────────────────────────────────────────

def classify(obsidian: list[dict], logos: list[dict], threshold: float):
    """Return (gaps, covered_set).
    gaps    = obsidian concepts with no close LOGOS match, sorted by source count
    covered = set of obsidian concept names that DO have a LOGOS match
    """
    gaps:    list[dict] = []
    covered: set[str]  = set()
    for obs in obsidian:
        best = max((cosine(obs["embedding"], l["embedding"]) for l in logos), default=0.0)
        if best >= threshold:
            covered.add(obs["concept"])
        else:
            gaps.append({**obs, "_best": best})
    gaps.sort(key=lambda x: x.get("sources", 1), reverse=True)
    return gaps, covered


# ── BFS path finding ──────────────────────────────────────────────────────────

def bfs_path(start: str, target: str,
             graph: dict[str, list[str]], max_depth: int = 4) -> list[str] | None:
    if start == target:
        return [start]
    queue   = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for nb in graph.get(path[-1], []):
            if nb == target:
                return path + [nb]
            if nb not in visited:
                visited.add(nb)
                queue.append(path + [nb])
    return None


def best_path_to_gap(gap_name: str, covered: set[str],
                     graph: dict[str, list[str]]) -> list[str] | None:
    best: list[str] | None = None
    for start in covered:
        p = bfs_path(start, gap_name, graph)
        if p and (best is None or len(p) < len(best)):
            best = p
    return best


# ── LLM skeleton ──────────────────────────────────────────────────────────────

def generate_skeleton(path: list[str], concept_data: dict[str, dict]) -> str:
    gap      = path[-1]
    evidence = ""
    for name in path:
        ins = concept_data.get(name, {}).get("insights", [])[:2]
        if ins:
            evidence += f"\n{name}:\n" + "".join(f"  • {i}\n" for i in ins)

    prompt = f"""You are helping an AI-native Product Manager write a LinkedIn post.

Concept path (what they know → new territory): {' -> '.join(path)}
New concept to introduce to readers: {gap}

Evidence from their personal knowledge graph:{evidence}

Write a tight post SKELETON only (not the full post):
  Hook: one counterintuitive or surprising sentence
  Point 1: one sentence
  Point 2: one sentence
  Point 3: one sentence
  Closing: one sentence CTA or insight

Under 100 words total. Direct PM practitioner voice. No buzzwords or hype."""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",         type=int,   default=3)
    parser.add_argument("--threshold",   type=float, default=0.40)
    parser.add_argument("--no-skeleton", action="store_true",
                        help="Skip LLM call — show paths and evidence only")
    args = parser.parse_args()

    # Validate deps
    for path, label in [(LOGOS_DB, "LOGOS KG — run: python kg/build.py"),
                        (OBS_CACHE, "Obsidian cache — run: python kg/compare.py")]:
        if not path.exists():
            print(f"Missing: {label}")
            return

    # Load
    print("Loading...")
    obsidian     = load_obsidian_cache()
    logos        = load_logos_concepts()
    concept_data = parse_all_concepts()

    graph = {name: data["related"] for name, data in concept_data.items()}
    graph = enrich_graph(graph, obsidian)   # fill sparse nodes via embeddings

    total_edges = sum(len(v) for v in graph.values())
    print(f"  Obsidian concepts : {len(obsidian)}")
    print(f"  LOGOS concepts    : {len(logos)}")
    print(f"  Concept graph     : {len(concept_data)} nodes, {total_edges} edges (incl. implicit)")

    gaps, covered = classify(obsidian, logos, args.threshold)
    print(f"  Covered by LOGOS  : {len(covered)}")
    print(f"  Gaps to bridge    : {len(gaps)}")
    print()

    # Find ideas: gaps that have a path from a covered concept
    ideas: list[dict] = []
    for gap in gaps:
        if len(ideas) >= args.top:
            break
        path = best_path_to_gap(gap["concept"], covered, graph)
        if path and len(path) >= 2:
            ideas.append({"gap": gap, "path": path})

    if not ideas:
        print("No paths found. Check Related Concepts are populated in 06_Concepts,")
        print("or lower --threshold.")
        return

    # ── Output ────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("WRITING PATH IDEAS")
    print("From concepts you've covered  ->  market gaps")
    print("=" * 65)

    for i, idea in enumerate(ideas, 1):
        gap  = idea["gap"]
        path = idea["path"]

        print(f"\nIDEA #{i}  |  {gap['concept']}  ({gap.get('sources', 1)} market sources)")
        print(f"  Path : {' -> '.join(path)}")
        print()
        print("  Evidence along the path:")
        for name in path:
            tag    = "  [you write about this]" if name in covered else ""
            ins    = concept_data.get(name, {}).get("insights", [])[:2]
            print(f"    [{name}]{tag}")
            for b in ins:
                print(f"      - {b[:90]}")
        print()

        if not args.no_skeleton:
            print("  SKELETON:")
            skeleton = generate_skeleton(path, concept_data)
            for line in skeleton.splitlines():
                print(f"    {line}")

        print()
        print("  " + "-" * 61)

    print(f"\nShowing {len(ideas)} of {len(gaps)} gap concepts with reachable paths.")
    print("Options:  --top N  |  --threshold 0.35  |  --no-skeleton")
    print("=" * 65)


if __name__ == "__main__":
    main()

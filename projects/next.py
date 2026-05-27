"""
LOGOS — projects/next.py
Scan concept graph + your experience log → suggest what to build next.

Logic:
  1. Find concepts you have My Experience in (experienced concepts)
  2. Find adjacent concepts (via Related Concepts + embedding similarity)
     that you have NOT yet experienced
  3. Rank by market importance (source count) + adjacency to your experience
  4. Suggest the top N as concrete build ideas

Usage:
    python projects/next.py            # top 3 build suggestions
    python projects/next.py --top 5
"""

import json
import re
import sys
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

ROOT           = Path(__file__).parent.parent
OBS_CACHE      = ROOT / "kg" / "obsidian_concepts_cache.json"
EXPERIENCE_LOG = ROOT / "projects" / "experience.jsonl"
CONCEPTS_DIR   = Path("G:/My Drive/AI Native PM/AI Native PM/06_Concepts")

client = OpenAI()


# ── Utilities ──────────────────────────────────────────────────────────────────

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_obsidian_concepts() -> list[dict]:
    if not OBS_CACHE.exists():
        print("Obsidian cache not found. Run: python kg/compare.py")
        sys.exit(1)
    return json.loads(OBS_CACHE.read_text(encoding="utf-8"))


def load_experience_log() -> list[dict]:
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


def parse_all_md() -> dict[str, dict]:
    """Parse all concept .md files.
    Returns: { canonical_name: { slug, related, insights, has_experience, experience_entries } }
    """
    slug_to_name: dict[str, str] = {}
    file_data:    dict[str, tuple[Path, str]] = {}

    for f in CONCEPTS_DIR.glob("*.md"):
        if f.name.startswith("_"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        slug = f.stem
        name = slug.replace("-", " ")
        if text.startswith("---"):
            for line in text.split("---", 2)[1].splitlines():
                if line.strip().startswith("concept:"):
                    name = line.split(":", 1)[1].strip()
                    break
        slug_to_name[slug] = name
        file_data[slug]    = (f, text)

    result: dict[str, dict] = {}
    for slug, (f, text) in file_data.items():
        name = slug_to_name[slug]

        # Related concepts
        related: list[str] = []
        rel_m = re.search(r"### Related Concepts\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL)
        if rel_m:
            for link in re.findall(r"\[\[([^\]|#]+)(?:\|[^\]]+)?\]\]", rel_m.group(1)):
                link = link.strip()
                related.append(slug_to_name.get(link, link.replace("-", " ")))

        # Key insights
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
                        if len(insights) >= 3:
                            break

        # My Experience
        exp_entries: list[str] = []
        exp_m = re.search(r"### My Experience\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL)
        if exp_m:
            for line in exp_m.group(1).splitlines():
                if line.strip().startswith("-"):
                    exp_entries.append(line.lstrip("- ").strip())

        result[name] = {
            "slug":           slug,
            "related":        related,
            "insights":       insights,
            "has_experience": bool(exp_entries),
            "exp_entries":    exp_entries,
        }
    return result


# ── Core logic ─────────────────────────────────────────────────────────────────

def find_experienced_concepts(concept_data: dict, log: list[dict]) -> set[str]:
    """Concepts that have a ### My Experience section OR appear in experience.jsonl."""
    experienced: set[str] = set()

    # From concept pages
    for name, data in concept_data.items():
        if data["has_experience"]:
            experienced.add(name)

    # From experience log (concept_matches field)
    for entry in log:
        for c in entry.get("concept_matches", []):
            experienced.add(c)

    return experienced


def score_candidate(name: str,
                    concept_data: dict,
                    concepts_by_name: dict,   # name -> obs cache dict
                    experienced: set[str],
                    experienced_embeddings: list[list[float]]) -> float:
    """
    Score an unexperienced concept as a next-build candidate.
    Higher = more worth building next.

    Factors:
      - market source count (from obsidian cache)
      - embedding proximity to your experienced concepts
      - whether it's explicitly related to experienced concepts
    """
    obs = concepts_by_name.get(name)
    if obs is None:
        return 0.0

    sources   = obs.get("sources", 1)
    related   = concept_data.get(name, {}).get("related", [])
    adj_bonus = sum(1 for r in related if r in experienced) * 0.15

    # Embedding proximity to experienced concepts
    if experienced_embeddings and obs.get("embedding"):
        proximity = max(cosine(obs["embedding"], e) for e in experienced_embeddings)
    else:
        proximity = 0.0

    return (sources * 0.4) + (proximity * 0.5) + adj_bonus


def generate_build_idea(candidate: str,
                        bridging_concepts: list[str],
                        concept_data: dict,
                        project_history: list[dict]) -> str:
    """LLM-generated concrete build idea."""
    insights = concept_data.get(candidate, {}).get("insights", [])[:3]
    history_text = ""
    for e in project_history[-3:]:  # last 3 entries
        history_text += f"- [{e['project']} {e.get('version','')}] {e['summary'][:80]}\n"

    bridge_text = ""
    for b in bridging_concepts:
        exp = concept_data.get(b, {}).get("exp_entries", [])
        if exp:
            bridge_text += f"\nYour experience with {b}:\n"
            for x in exp[:2]:
                bridge_text += f"  - {x[:80]}\n"

    prompt = f"""You are advising an AI-native Product Manager on what to build next.

Target concept to implement: {candidate}
Key insights about this concept:
{chr(10).join(f'- {i}' for i in insights)}

Related concepts you've already experienced:{bridge_text if bridge_text else ' (none yet)'}

Recent project history:
{history_text if history_text else '(no history yet)'}

Suggest ONE concrete, specific build idea that would give this PM hands-on experience
with "{candidate}". The idea should:
- Be completable as a side project in 1-3 weeks
- Build directly on their existing projects (MentorFlow, PM_OS tools, LOGOS itself)
- Produce a tangible artifact (feature, script, experiment, documented finding)

Format:
Build: [one sentence describing what to build]
Why: [one sentence connecting to their existing work]
Artifact: [what they'll have when done]

Under 80 words total."""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",         type=int,   default=3)
    parser.add_argument("--no-idea",     action="store_true",
                        help="Skip LLM — show candidates only")
    args = parser.parse_args()

    if not OBS_CACHE.exists():
        print("Obsidian cache not found. Run: python kg/compare.py")
        return

    # Load data
    print("Loading...")
    obsidian     = load_obsidian_concepts()
    concept_data = parse_all_md()
    log          = load_experience_log()

    concepts_by_name = {c["concept"]: c for c in obsidian}

    # Find experienced concepts
    experienced = find_experienced_concepts(concept_data, log)

    if not experienced:
        print("\nNo experience logged yet.")
        print("Run: python projects/log.py")
        print("\nTo get started, log experience from any recent project or PM work.")
        return

    print(f"  Concepts with your experience : {len(experienced)}")
    print(f"  Total concepts                : {len(concept_data)}")
    print(f"  Experience log entries        : {len(log)}")
    print()

    # Get embeddings of experienced concepts for proximity scoring
    experienced_embeddings = [
        concepts_by_name[n]["embedding"]
        for n in experienced
        if n in concepts_by_name and concepts_by_name[n].get("embedding")
    ]

    # Score all unexperienced concepts
    candidates = []
    for name in concept_data:
        if name in experienced:
            continue
        score = score_candidate(
            name, concept_data, concepts_by_name,
            experienced, experienced_embeddings
        )
        # Find bridging concepts (adjacent experienced concepts)
        related    = concept_data[name].get("related", [])
        bridges    = [r for r in related if r in experienced]
        candidates.append((score, name, bridges))

    candidates.sort(reverse=True)
    top_candidates = candidates[:args.top]

    # Output
    print("=" * 65)
    print("NEXT BUILD IDEAS")
    print("Based on your experience graph")
    print("=" * 65)

    print("\nYour experienced concepts:")
    for name in sorted(experienced):
        entries = concept_data.get(name, {}).get("exp_entries", [])
        tag = f"  [{entries[0][:55]}]" if entries else ""
        print(f"  + {name}{tag}")
    print()

    for i, (score, name, bridges) in enumerate(top_candidates, 1):
        obs     = concepts_by_name.get(name, {})
        sources = obs.get("sources", 1)

        print(f"{'─' * 65}")
        print(f"IDEA #{i}  |  {name}  [{sources} market sources]")
        if bridges:
            print(f"  Bridge from: {', '.join(bridges)}")

        # Theory
        insights = concept_data.get(name, {}).get("insights", [])[:2]
        if insights:
            print("  What the market says:")
            for ins in insights:
                print(f"    - {ins[:90]}")

        if not args.no_idea and log:
            print()
            print("  BUILD IDEA:")
            idea = generate_build_idea(name, bridges, concept_data, log)
            for line in idea.splitlines():
                print(f"    {line}")
        elif not log:
            print("  (Log some experience first to get personalised build ideas)")

        print()

    print("=" * 65)
    print("Log new experience: python projects/log.py")
    print("Write about it:     python kg/inspire.py")


if __name__ == "__main__":
    main()

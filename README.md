# LOGOS

Personal knowledge publishing layer for an AI-native PM.

λόγος (logos) — word, reason, argument. The system that turns accumulated knowledge into published positions.

## What this is

LOGOS is a lightweight Python toolset that connects three things:

1. **A personal knowledge graph** (Obsidian `06_Concepts` — 57 concepts extracted from 30+ AI PM sources via PRAXIS)
2. **A LinkedIn publishing workflow** (23 published posts, brand-validated)
3. **A project experience log** (side projects + day-job milestones mapped to concepts)

The flywheel:

```
PRAXIS (external knowledge ingestion)
    ↓
Obsidian 06_Concepts (57 concept nodes)
    ↓                          ↑
kg/compare.py ──→ gaps    projects/log.py (personal experience)
    ↓                          ↓
kg/inspire.py ──→ writing paths + skeletons
    ↓
LOGOS posts (check.py → publish → add.py)
    ↓
projects/next.py ──→ next build idea
```

## Tools

### Pre/post publish

```bash
python check.py       # brand consistency check before publishing
python add.py         # index post after publishing, auto-updates SYSTEM_STATUS.md
```

### Knowledge graph (`kg/`)

```bash
python kg/build.py              # extract concepts from LPL posts → logos_kg.db
python kg/compare.py            # brand vs. market gap analysis (Obsidian 06_Concepts)
python kg/compare.py --source praxis   # compare against PRAXIS halos.db instead
python kg/inspire.py            # traverse concept graph → writing path ideas + skeletons
python kg/inspire.py --no-skeleton     # fast mode, no LLM call
```

### Project experience (`projects/`)

```bash
python projects/log.py          # log a project experience → maps to concepts → writes My Experience
python projects/log.py --list   # show all logged experiences
python projects/next.py         # suggest next build idea based on concept graph + experience
python projects/next.py --no-idea     # fast mode, no LLM call
```

## The knowledge loop

1. **External knowledge** enters via PRAXIS (video/audio ingestion) → distilled into `06_Concepts`
2. **`kg/compare.py`** finds what the market discusses that you haven't written about
3. **`kg/inspire.py`** finds a path from concepts you've written about → to the gap
4. **`projects/log.py`** maps your personal experience onto concepts (adds `### My Experience` to concept pages)
5. **`projects/next.py`** recommends what to build next — grounded in your experience graph, not random AI suggestions
6. After building → log it → write about it → posts have first-hand authority

## Concept graph

57 concept nodes, sourced from 30+ AI PM talks and articles. Top concepts by market signal:

| Concept | Sources |
|---------|---------|
| AI Product Development Speed | 17 |
| Agency in Product Development | 9 |
| PM Role Evolution | 9 |
| Agency-Autonomy Tradeoff | 6 |
| Agent Memory | 4 |
| Distribution as Moat | 4 |
| Product Taste | 4 |

## Brand positioning

Richard Lee — AI-native PM. Dublin, Ireland.

Core belief: AI accelerates capability, but accountability stays with the person.

Four content clusters:
- **A** — AI widens the judgment gap (6 posts)
- **B** — PM defines boundaries in ambiguity (7 posts)
- **C** — Build-in-public (4 posts)
- **D** — AI as infrastructure (6 posts)

## Setup

```bash
pip install openai python-dotenv rapidfuzz numpy
cp config/local_paths.sample.json config/local_paths.json
# edit local_paths.json: set obsidian_vault_root to your vault path
# create .env: OPENAI_API_KEY=sk-...
```

## Structure

```
LOGOS/
  check.py                    # pre-publish brand check
  add.py                      # post-publish indexer
  brand_context.json          # positioning statement + cluster definitions
  SYSTEM_STATUS.md            # auto-updated snapshot
  config/
    local_paths.sample.json
  kg/
    build.py                  # LPL posts → concept KG
    compare.py                # brand vs market gap analysis
    inspire.py                # concept graph → writing paths
  projects/
    log.py                    # experience → concept pages
    next.py                   # concept graph → next build idea
  prompts/                    # LLM prompt templates
```

## Ecosystem

LOGOS is one layer in a broader personal knowledge system:

| System | Role |
|--------|------|
| PRAXIS | Ingests external knowledge (video/audio → KG) |
| Obsidian / AI Native PM | Stores synthesised concepts (06_Concepts) |
| **LOGOS** | Publishes knowledge as LinkedIn posts |
| PM_OS | Execution layer (current work context) |
| JobSeekerExpert | Market signal (JD analysis) |

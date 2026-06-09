"""
LOGOS — integrity.py
Integrity check across three sources of truth:
  1. lpl_index.jsonl   — master index
  2. 08_LPL_Library    — .md files on disk
  3. SYSTEM_STATUS.md  — cluster counts snapshot

Usage:
    python integrity.py          # report only
    python integrity.py --fix    # report + auto-fix SYSTEM_STATUS counts
"""

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

ROOT   = Path(__file__).parent
CONFIG = json.loads((ROOT / "config" / "local_paths.json").read_text(encoding="utf-8"))
VAULT  = Path(CONFIG["obsidian_vault_root"])
INDEX  = VAULT / CONFIG["vault_paths"]["agent_data_dir"] / "lpl_index.jsonl"
LPL_DIR = VAULT / CONFIG["vault_paths"]["lpl_dir"]
STATUS_PATH = ROOT / "SYSTEM_STATUS.md"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_index() -> list[dict]:
    if not INDEX.exists():
        return []
    entries = []
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def all_library_ids() -> dict[str, Path]:
    """Return {lpl_id: path} for every LPL-*.md found under LPL_DIR."""
    result = {}
    if not LPL_DIR.exists():
        return result
    for md in LPL_DIR.rglob("LPL-*.md"):
        # Extract lpl_id from filename (strip long suffix titles if any)
        stem = md.stem
        lpl_id = stem.split("_")[0] if "_" in stem else stem
        result[lpl_id] = md
    return result


def count_clusters(entries: list[dict]) -> dict[str, int]:
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for e in entries:
        c = e.get("cluster", "")
        if c in counts:
            counts[c] += 1
    return counts


def parse_status_counts() -> dict[str, int]:
    """Read current cluster counts from SYSTEM_STATUS.md."""
    counts = {}
    if not STATUS_PATH.exists():
        return counts
    for line in STATUS_PATH.read_text(encoding="utf-8").splitlines():
        for letter in ("A", "B", "C", "D"):
            if f"| Cluster {letter}" in line:
                parts = [p.strip() for p in line.split("|")]
                # parts: ['', 'Cluster X (...)', 'count', '']
                try:
                    counts[letter] = int(parts[2])
                except (IndexError, ValueError):
                    pass
    return counts


def fix_system_status(counts: dict[str, int], total: int) -> None:
    """Rewrite SYSTEM_STATUS.md snapshot block with correct counts."""
    today = datetime.now().strftime("%Y-%m-%d")
    text = STATUS_PATH.read_text(encoding="utf-8")
    replacements = {
        "| Posts published |":  f"| Posts published | {total} |",
        "| Cluster A":          f"| Cluster A (AI widens judgment gap) | {counts['A']} |",
        "| Cluster B":          f"| Cluster B (PM defines boundaries) | {counts['B']} |",
        "| Cluster C":          f"| Cluster C (Build-in-public) | {counts['C']} |",
        "| Cluster D":          f"| Cluster D (AI as infrastructure) | {counts['D']} |",
        "| Last activity |":    f"| Last activity | {today} |",
    }
    new_lines = []
    for line in text.splitlines():
        replaced = False
        for prefix, new_val in replacements.items():
            if line.strip().startswith(prefix):
                new_lines.append(new_val)
                replaced = True
                break
        if not replaced:
            if line.startswith("**Updated:**"):
                new_lines.append(f"**Updated:** {today}  **Status:** Active")
            else:
                new_lines.append(line)
    STATUS_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# ── main check ───────────────────────────────────────────────────────────────

def run_check(fix: bool = False, silent_if_clean: bool = False) -> bool:
    """
    Run integrity check. Returns True if clean, False if issues found.
    fix=True  → auto-fix SYSTEM_STATUS counts if they're wrong.
    silent_if_clean=True → print nothing if no issues (for use inside add.py).
    """
    issues: list[str] = []

    entries   = load_index()
    lib_ids   = all_library_ids()
    index_ids = {e["lpl_id"]: e for e in entries}

    # ── 1. Index entries missing .md file ────────────────────────────────────
    missing_md = []
    for e in entries:
        lpl_id = e["lpl_id"]
        path_field = e.get("path", "")
        if path_field:
            expected = VAULT / path_field
            if not expected.exists():
                missing_md.append((lpl_id, str(expected)))
        elif lpl_id not in lib_ids:
            missing_md.append((lpl_id, "(no path field, not found in library)"))

    if missing_md:
        issues.append("Index entries with no .md file in library:")
        for lpl_id, path in missing_md:
            issues.append(f"  [!]{lpl_id}  →  {path}")

    # ── 2. Library .md files missing from index ───────────────────────────────
    orphan_md = [lpl_id for lpl_id in lib_ids if lpl_id not in index_ids]
    if orphan_md:
        issues.append("Library .md files with no index entry:")
        for lpl_id in sorted(orphan_md):
            issues.append(f"  [!]{lpl_id}")

    # ── 3. SYSTEM_STATUS cluster counts ──────────────────────────────────────
    actual_counts = count_clusters(entries)
    actual_total  = sum(actual_counts.values())
    status_counts = parse_status_counts()

    count_mismatches = []
    for letter in ("A", "B", "C", "D"):
        actual  = actual_counts[letter]
        in_file = status_counts.get(letter)
        if in_file != actual:
            count_mismatches.append((letter, in_file, actual))

    if count_mismatches:
        issues.append("SYSTEM_STATUS.md cluster counts are out of sync:")
        for letter, in_file, actual in count_mismatches:
            issues.append(f"  [!]Cluster {letter}: SYSTEM_STATUS={in_file}  index={actual}")

    # ── report ────────────────────────────────────────────────────────────────
    if not issues:
        if not silent_if_clean:
            print("[OK] Integrity check passed -- index, library, and SYSTEM_STATUS are in sync.")
        return True

    print("-- LOGOS integrity check -----------------------------")
    for line in issues:
        print(line)
    print("------------------------------------------------------")

    if fix:
        if count_mismatches:
            fix_system_status(actual_counts, actual_total)
            print(f"[FIXED] SYSTEM_STATUS.md updated  (total={actual_total}, "
                  f"A={actual_counts['A']}, B={actual_counts['B']}, "
                  f"C={actual_counts['C']}, D={actual_counts['D']})")
        if missing_md or orphan_md:
            print("[WARN] File-level issues require manual review (not auto-fixed).")

    return False


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LOGOS integrity check")
    parser.add_argument("--fix", action="store_true",
                        help="Auto-fix SYSTEM_STATUS.md if counts are wrong")
    args = parser.parse_args()
    ok = run_check(fix=args.fix)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

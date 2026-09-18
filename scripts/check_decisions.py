#!/usr/bin/env python3
"""Does every decision in docs/DECISIONS.md still point at evidence that exists?

The table is the registry. For each D-row: every file named in the evidence
and recorded-in cells must exist (abbreviations such as `PLAN-exit-priced`
resolve by prefix), every cited section (`§3`, `§2.1`, `§1–§3`) must be a
heading in that file, and every backticked sha must be in git. Exit 1 on any
miss. Run from anywhere: `python scripts/check_decisions.py`.

Not caught: a section that still exists but no longer says what the decision
relied on. That needs a reader, which is what the verifier's pass is for.
"""
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH = [ROOT / "experiments/v2/MZ01_maze_world", ROOT / "experiments/v2/MZ01_maze_world/record",
          ROOT / "docs", ROOT / "docs/planning", ROOT / "scripts/box", ROOT / "src/calibration", ROOT]
FILE = re.compile(r"`?((?:[\w./-]+/)?(?:PLAN-[\w-]+|RESULTS-[\w-]+|REPORT-[\w-]+|DIAGNOSIS-[\w-]+|SPEC|README|[A-Z_]+_PLAN)(?:\.md|\.py|\.sh)?|[\w-]+\.(?:md|py|sh))`?\s*((?:§\s*[\dA-Z]+(?:\.\d+)?(?:\s*[–-]\s*§?\s*[\dA-Z]+(?:\.\d+)?)?(?:,\s*)?)*)")
SHA = re.compile(r"`([0-9a-f]{7,40})`")
SEC = re.compile(r"§\s*([\dA-Z]+(?:\.\d+)?)")


def resolve(name):
    stem = Path(name).name
    for d in SEARCH:
        hits = sorted(p for p in d.glob(stem.split(".")[0] + "*") if p.suffix in (".md", ".py", ".sh"))
        if hits:
            return hits[0]
    return None


def has_heading(path, sec):
    pat = re.compile(r"^#{1,4}\s*(?:§\s*)?" + re.escape(sec) + r"[.\s)]")
    return any(pat.match(line) for line in path.read_text(errors="replace").splitlines())


def check(row):
    cells = [c.strip() for c in row.strip("|").split("|")]
    did, evidence, recorded = cells[0], cells[3], cells[4]
    problems, refs = [], 0
    for text in (evidence, recorded):
        for name, secs in FILE.findall(text):
            path = resolve(name)
            refs += 1
            if path is None:
                problems.append(f"no file for {name!r}")
                continue
            for sec in SEC.findall(secs):
                refs += 1
                if not has_heading(path, sec):
                    problems.append(f"{path.relative_to(ROOT)} has no §{sec}")
        for sha in SHA.findall(text):
            refs += 1
            if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", sha], capture_output=True).returncode:
                problems.append(f"sha {sha} not in git")
    return did, refs, problems


def main():
    rows = [l for l in (ROOT / "docs/DECISIONS.md").read_text().splitlines() if l.startswith("| D-")]
    bad = total = 0
    for row in rows:
        did, refs, problems = check(row)
        total += refs
        print(f"[{'ok ' if not problems else 'BAD'}] {did} ({refs} refs)" + "".join(f"\n      ! {p}" for p in problems))
        bad += bool(problems)
    print(f"{len(rows)} decisions, {total} references checked, {bad} decisions with problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# seed_placeholders.sh [--print] [staging_dir]
#
# The puller (scripts/pull_from_sandbox.sh) skips any remote file it already
# has locally at the same size, and takes at most ONE >1MB file per cycle at
# roughly a minute per megabyte. This lease's snapshot carries 115 pullable
# files over 1 MB totalling ~1.49 GB -- 112 of them HH01's adapters from
# 2026-08-27, none of them this task's output. Left alone a watcher would spend
# about a DAY dragging them down. Size-matched (sparse, so zero disk)
# placeholders in the staging dir make the puller skip them.
#
# Placeholdered:
#   * every pullable file > $THRESH that is NOT MZ01 output;
#   * the three MZ01 files this session PUSHED (they are inputs; we have them);
#   * any results/orgb/orgb_ckpt_*.pt at any size -- a 126 MB mid-cell
#     checkpoint rewritten every 50 steps pulls torn and still passes the
#     puller's size check. Rescuing one is a deliberate act, never background.
# Never placeholdered: anything else under MZ01_maze_world/results -- that is
# the trial's own output and must come home.
set -euo pipefail

SP="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding ${BOX:-box1}.url and ${BOX:-box1}.token}"
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ex.sh"
THRESH="${PLACEHOLDER_THRESH:-1000000}"

PRINT=0
if [ "${1:-}" = "--print" ]; then PRINT=1; shift; fi
DEST="${1:-$SP/pull_${BOX:-box1}}"

MANIFEST="$SP/placeholders.tsv"
: > "$MANIFEST.new"

python3 - "$THRESH" <<'PY' | bash "$EX" - 2>/dev/null | grep -v '^Warning: connecting' > "$SP/.placeholder_list.txt"
import sys
thresh = int(sys.argv[1])
print(f'''import glob, os
THRESH = {thresh}
KEEP_INPUTS = {{
  "experiments/v2/MZ01_maze_world/results/navigator/nav_nav48.pt",
  "experiments/v2/MZ01_maze_world/results/navigator/nav_nav48.json",
  "experiments/v2/MZ01_maze_world/results/orgb/warm_start.pt",
}}
seen, rows = set(), []
for pat in ["/marimo/repo/experiments/**/results*/**",
            "/marimo/repo/*_status.txt", "/marimo/repo/*_driver.log"]:
    for p in glob.glob(pat, recursive=True):
        if os.path.isfile(p) and p not in seen:
            seen.add(p); rows.append(p)
for p in sorted(rows):
    rel = p[len("/marimo/repo/"):]
    size = os.path.getsize(p)
    mz01 = rel.startswith("experiments/v2/MZ01_maze_world/results/")
    ckpt = mz01 and "/orgb/orgb_ckpt_" in rel and rel.endswith(".pt")
    if ckpt:
        why = "midcell-checkpoint"
    elif mz01 and rel in KEEP_INPUTS:
        why = "pushed-input"
    elif mz01:
        continue                      # this task's output: always pull it
    elif size > THRESH:
        why = "foreign-bulk"
    else:
        continue
    print("P", size, why, rel)''')
PY

N=$(grep -c '^P ' "$SP/.placeholder_list.txt" || true)
BYTES=$(awk '$1=="P"{s+=$2} END{print s+0}' "$SP/.placeholder_list.txt")
if [ "$N" = "0" ]; then
  echo "SEED_ABORT the box returned no candidates -- is it alive?" >&2
  exit 3
fi
echo "candidates: $N files, $BYTES bytes  (threshold >$THRESH B)"
awk '$1=="P"{print "  " $3 "  " $2 "  " $4}' "$SP/.placeholder_list.txt" | head -8
echo "  ... (full list: $SP/.placeholder_list.txt)"

if [ "$PRINT" = "1" ]; then
  echo "--- would create sparse size-matched placeholders under $DEST ---"
  echo "--- ($N files; nothing written; rerun without --print to seed) ---"
  rm -f "$MANIFEST.new"
  exit 0
fi

made=0; kept=0; clash=0
while read -r _p size why rel; do
  lp="$DEST/$rel"
  if [ -e "$lp" ]; then
    have=$(wc -c < "$lp")
    if [ "$have" = "$size" ]; then kept=$((kept+1)); else
      echo "  CLASH $rel local=$have remote=$size (left alone; delete it by hand if it is junk)" >&2
      clash=$((clash+1)); fi
    continue
  fi
  mkdir -p "$(dirname "$lp")"
  truncate -s "$size" "$lp"
  printf '%s\t%s\t%s\n' "$size" "$why" "$rel" >> "$MANIFEST.new"
  made=$((made+1))
done < <(grep '^P ' "$SP/.placeholder_list.txt")

mv "$MANIFEST.new" "$MANIFEST"
date -u +%Y-%m-%dT%H:%M:%SZ > "$DEST/.placeholders-seeded"
echo "SEEDED made=$made already_matched=$kept clashes=$clash under $DEST"
echo "manifest: $MANIFEST   disk used: $(du -sh "$DEST" | cut -f1) (sparse)"

#!/usr/bin/env bash
# pull_once.sh [--print] <label>
#
# One mirror cycle out of box 1 into $SP/pull_${BOX:-box1}, using the repo's own
# scripts/pull_from_sandbox.sh (MARIMO_EXEC / URL / TOKEN are the env it wants;
# TOKEN is the token VALUE, not a file). Nothing is written into the worktree.
#
# <label> only names the log ($SP/pull_<label>.log) -- pull_from_sandbox.sh has
# no per-run filter, it mirrors experiments/**/results*, *_status.txt and
# *_driver.log wholesale. Run seed_placeholders.sh FIRST or this drags down the
# snapshot's foreign adapters at ~1 min/MB.
#
# NOTE: pull_from_sandbox.sh passes the token as --token, so it is visible in
# `ps` on this machine for the life of each call. That is the repo script's
# interface, not a choice made here.
set -euo pipefail

SP="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding ${BOX:-box1}.url and ${BOX:-box1}.token}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXEC_SH="$WT/.claude/skills/marimo-pair/scripts/execute-code.sh"
PULLER="$WT/scripts/pull_from_sandbox.sh"

PRINT=0
if [ "${1:-}" = "--print" ]; then PRINT=1; shift; fi

LABEL="${1:?usage: pull_once.sh [--print] <label>}"
DEST="$SP/pull_${BOX:-box1}"
LOG="$SP/pull_${LABEL}.log"

if [ "$PRINT" = "1" ]; then
  echo "--- pull_once.sh would run ---"
  echo "MARIMO_EXEC=$EXEC_SH \\"
  echo "URL=$(cat "$SP/${BOX:-box1}.url") \\"
  echo "TOKEN=<\$SP/${BOX:-box1}.token, 64 chars, not printed> \\"
  echo "  bash $PULLER $DEST --once"
  echo "--- tee-ing to $LOG; placeholders seeded: $([ -e "$DEST/.placeholders-seeded" ] && cat "$DEST/.placeholders-seeded" || echo NO) ---"
  exit 0
fi

mkdir -p "$DEST"
echo "PULL_ONCE label=$LABEL dest=$DEST log=$LOG  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ ! -e "$DEST/.placeholders-seeded" ]; then
  echo "WARNING: no placeholders in $DEST -- run 'bash $SP/seed_placeholders.sh' first" >&2
fi

MARIMO_EXEC="$EXEC_SH" \
URL="$(cat "$SP/${BOX:-box1}.url")" \
TOKEN="$(cat "$SP/${BOX:-box1}.token")" \
  bash "$PULLER" "$DEST" --once 2>&1 | tee -a "$LOG"

echo "PULL_ONCE done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
find "$DEST/experiments/v2/MZ01_maze_world" -type f -newermt '-1 hour' 2>/dev/null | sed 's/^/  fresh: /' || true

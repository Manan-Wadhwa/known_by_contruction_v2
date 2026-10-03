#!/usr/bin/env bash
# Claude Code hook that keeps HANDOFF.md current. Usage: handoff-hook.sh start|stop,
# with the hook's JSON on stdin (session_id, cwd, stop_hook_active).
#   start: prints HANDOFF.md into the session's context and saves a baseline
#          (HEAD, a hash of the tracked changes outside HANDOFF.md, a hash of
#          HANDOFF.md) in <git dir>/handoff-<session_id>, unless one exists,
#          so compaction and resume keep the session's first baseline.
#   stop:  if HEAD or the tracked changes moved since the baseline and HANDOFF.md
#          did not, prints a block decision; if HANDOFF.md moved, the baseline
#          moves forward. Untracked files never count (box pulls land untracked).
# Exits 0 quietly outside git, without jq, or without a session_id.

command -v jq >/dev/null 2>&1 || exit 0
input=$(cat)
dir=$(jq -r '.cwd // empty' <<<"$input" 2>/dev/null)
cd "${dir:-.}" 2>/dev/null || exit 0
root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$root" || exit 0
sid=$(jq -r '.session_id // empty' <<<"$input" 2>/dev/null)
[ -n "$sid" ] || exit 0
base="$(git rev-parse --absolute-git-dir)/handoff-$sid"

snapshot() {
  # One line, "HEAD changes-hash handoff-hash", for the tree as it is now.
  local head changes handoff
  head=$(git rev-parse -q --verify HEAD 2>/dev/null)
  changes=$(git diff HEAD --binary -- . ':(exclude)HANDOFF.md' 2>/dev/null | sha1sum | cut -d' ' -f1)
  handoff=$(sha1sum HANDOFF.md 2>/dev/null | cut -d' ' -f1)
  echo "$head $changes $handoff"
}

case "$1" in
  start)
    [ -f HANDOFF.md ] && { echo "== HANDOFF.md (live state; update your ## section before you stop) =="; cat HANDOFF.md; }
    [ -f "$base" ] || snapshot >"$base"
    ;;
  stop)
    [ "$(jq -r '.stop_hook_active // false' <<<"$input")" = true ] && exit 0
    now=$(snapshot)
    [ -f "$base" ] || { echo "$now" >"$base"; exit 0; }
    read -r b_head b_changes b_handoff <"$base"
    read -r n_head n_changes n_handoff <<<"$now"
    if [ "$n_handoff" != "$b_handoff" ]; then
      echo "$now" >"$base"
    elif [ "$n_head" != "$b_head" ] || [ "$n_changes" != "$b_changes" ]; then
      jq -n '{decision: "block", reason: "You changed the repo this session but HANDOFF.md is unchanged. Update your ## <owner> section (in flight, last verdicts, next, gotchas), commit it, then stop."}'
    fi
    ;;
esac
exit 0

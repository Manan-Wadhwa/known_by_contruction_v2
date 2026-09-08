#!/usr/bin/env bash
# launch_trial_on_box.sh [--print] <label> <log_path> KEY=VAL [KEY=VAL ...]
#
# Writes /marimo/repo/<label>_launch.sh on box 1 and starts it detached
# (setsid + nohup) so it outlives the exec channel, then prints the pid.
#
#   * every KEY=VAL becomes an `export` in the launcher; values are shell-quoted
#     locally, so spaces and empty values (MZ01_TRIAL_CELL_2= ) survive intact;
#   * CALIBRATION_GIT_SHA must be one of the KEY=VALs. It is refused if absent,
#     because runner.git_sha() falls back to it and a manifest that records a
#     sha nobody set is a manifest that lies about which code ran;
#   * it REFUSES to start if this label's launcher, or any mz01_explore_trial.py,
#     is already running on the box -- two trials sharing /marimo/repo would
#     interleave their writes to mz01_explore_status.txt;
#   * --print writes nothing anywhere: it prints the launcher it would install
#     and the command it would run, and exits.
set -euo pipefail

SP="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding ${BOX:-box1}.url and ${BOX:-box1}.token}"
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ex.sh"
TRIAL="scripts/mz01_explore_trial.py"

PRINT=0
if [ "${1:-}" = "--print" ]; then PRINT=1; shift; fi

LABEL="${1:?usage: launch_trial_on_box.sh [--print] <label> <log_path> KEY=VAL ...}"
LOGPATH="${2:?usage: launch_trial_on_box.sh [--print] <label> <log_path> KEY=VAL ...}"
shift 2
[ $# -gt 0 ] || { echo "LAUNCH_REFUSED no KEY=VAL environment given" >&2; exit 2; }

case "$LABEL" in
  *[!A-Za-z0-9_-]*|"") echo "LAUNCH_REFUSED label must be [A-Za-z0-9_-]+: '$LABEL'" >&2; exit 2 ;;
esac

SAW_SHA=0
for kv in "$@"; do
  case "$kv" in
    *=*) : ;;
    *) echo "LAUNCH_REFUSED not a KEY=VAL: '$kv'" >&2; exit 2 ;;
  esac
  [ "${kv%%=*}" = "CALIBRATION_GIT_SHA" ] && SAW_SHA=1
done
if [ "$SAW_SHA" != "1" ]; then
  echo "LAUNCH_REFUSED CALIBRATION_GIT_SHA=<sha> must be among the KEY=VALs" >&2
  exit 2
fi

REMOTE_SH="/marimo/repo/${LABEL}_launch.sh"
REMOTE_PID="/marimo/repo/${LABEL}.pid"

LAUNCHER=$(python3 - "$LABEL" "$REMOTE_PID" "$TRIAL" "$@" <<'PY'
import shlex, sys
label, pidfile, trial = sys.argv[1], sys.argv[2], sys.argv[3]
kvs = sys.argv[4:]
lines = ["#!/bin/bash",
         f"# {label} -- written by launch_trial_on_box.sh; do not hand-edit",
         "set -u",
         "cd /marimo/repo || exit 91"]
for kv in kvs:
    k, _, v = kv.partition("=")
    lines.append(f"export {k}={shlex.quote(v)}")
lines += [f"echo $$ > {shlex.quote(pidfile)}",
          "date -u +'launched %Y-%m-%dT%H:%M:%SZ'",
          f"exec python3 {trial}"]
print("\n".join(lines))
PY
)

if [ "$PRINT" = "1" ]; then
  echo "--- would write $REMOTE_SH on the box ---"
  printf '%s\n' "$LAUNCHER"
  echo "--- would then run, on the box ---"
  echo "setsid nohup bash $REMOTE_SH > $LOGPATH 2>&1 < /dev/null &"
  echo "--- and report the pid from $REMOTE_PID ---"
  exit 0
fi

# --- refuse a double launch ------------------------------------------------
BUSY=$(python3 - "$LABEL" <<'PY' | bash "$EX" - 2>/dev/null | sed -n 's/^BUSY //p'
import sys
label = sys.argv[1]
print(f'''import subprocess
mine = subprocess.run("pgrep -af {label}_launch.sh | grep -v pgrep", shell=True, capture_output=True, text=True).stdout.strip()
any_ = subprocess.run("pgrep -af mz01_explore_trial.py | grep -v pgrep", shell=True, capture_output=True, text=True).stdout.strip()
print("BUSY", ("label:" + mine.replace(chr(10), " | ")) if mine else ("other:" + any_.replace(chr(10), " | ")) if any_ else "no")''')
PY
)
if [ -z "$BUSY" ]; then
  echo "LAUNCH_REFUSED could not reach the box to check for a running trial" >&2
  exit 3
fi
if [ "$BUSY" != "no" ]; then
  echo "LAUNCH_REFUSED something is already running: $BUSY" >&2
  exit 4
fi

# --- install and start -----------------------------------------------------
B64=$(printf '%s\n' "$LAUNCHER" | base64 -w0)
python3 - "$B64" "$REMOTE_SH" "$LOGPATH" "$REMOTE_PID" <<'PY' | bash "$EX" - 2>&1 | grep -v '^Warning: connecting'
import sys
b64, rsh, logp, pidf = sys.argv[1:5]
print(f'''import base64, os, subprocess, time, datetime
open({rsh!r}, "wb").write(base64.b64decode("{b64}"))
os.chmod({rsh!r}, 0o755)
os.makedirs(os.path.dirname({logp!r}) or "/", exist_ok=True)
if os.path.exists({pidf!r}):
    os.remove({pidf!r})
print("LAUNCHER_SHA", __import__("hashlib").sha256(open({rsh!r},"rb").read()).hexdigest()[:16])
print("LAUNCH_UTC", datetime.datetime.now(datetime.timezone.utc).isoformat())
subprocess.run("setsid nohup bash {rsh} > {logp} 2>&1 < /dev/null &", shell=True, start_new_session=True)
for _ in range(30):
    time.sleep(1)
    if os.path.exists({pidf!r}):
        break
pid = open({pidf!r}).read().strip() if os.path.exists({pidf!r}) else ""
print("PID", pid or "UNKNOWN")
print("ALIVE", os.path.exists("/proc/" + pid) if pid else False)
print("LOG_HEAD")
print(subprocess.run("tail -20 {logp}", shell=True, capture_output=True, text=True).stdout)''')
PY

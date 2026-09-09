#!/usr/bin/env bash
# Targeted pull: only the trial's own files, base64 over the exec channel,
# sha256-verified, mirrored into the worktree's results dir (gitignored).
# Usage: pull_trial.sh <label>   e.g. exitcap_cellA_s0
set -euo pipefail
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding ${BOX:-box1}.url and ${BOX:-box1}.token}"
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.claude/skills/marimo-pair/scripts/execute-code.sh"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LABEL=$1
DEST=$WT/experiments/v2/MZ01_maze_world/results/$LABEL
mkdir -p "$DEST"
bash "$EX" --url "$(cat $S/${BOX:-box1}.url)" --token "$(cat $S/${BOX:-box1}.token)" - <<PY 2>/dev/null > $S/.pull_$LABEL.json
import base64, hashlib, json, os, glob
d = "/marimo/repo/experiments/v2/MZ01_maze_world/results/$LABEL"
# The trial's status file is box-wide (one path at the repo root), so a box that ran two jobs
# has one file; copying it into every run directory put another job's header there. The run's own log carries everything it did.
paths = sorted(glob.glob(d + "/*")) + [p for p in ("/marimo/repo/$LABEL.log",) if os.path.exists(p)]
out = []
for p in paths:
    if os.path.isdir(p): continue
    b = open(p, "rb").read()
    out.append({"name": os.path.basename(p), "sha": hashlib.sha256(b).hexdigest(), "n": len(b), "b64": base64.b64encode(b).decode()})
print("PULLJSON" + json.dumps(out))
PY
python3 - "$DEST" "$S/.pull_$LABEL.json" <<'PY'
import base64, hashlib, json, sys, os
dest, src = sys.argv[1], sys.argv[2]
txt = open(src).read()
i = txt.find("PULLJSON")
if i < 0:
    print("PULL FAILED: no payload", txt[-300:]); sys.exit(1)
items = json.loads(txt[i + 8:].strip().splitlines()[0])
for it in items:
    b = base64.b64decode(it["b64"])
    if hashlib.sha256(b).hexdigest() != it["sha"] or len(b) != it["n"]:
        print("SHA MISMATCH", it["name"]); sys.exit(1)
    tmp = os.path.join(dest, it["name"] + ".part")
    open(tmp, "wb").write(b); os.replace(tmp, os.path.join(dest, it["name"]))
print("PULL_OK", len(items), "files:", " ".join(f"{it['name']}({it['n']})" for it in items))
PY

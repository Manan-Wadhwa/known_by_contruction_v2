#!/usr/bin/env bash
# pull_files.sh <box> <local_dir> <remote_path>...  -- base64 over the channel, sha256-checked
set -euo pipefail
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding boxN.url and boxN.token}"
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.claude/skills/marimo-pair/scripts/execute-code.sh"
BOX=$1; DEST=$2; shift 2; mkdir -p "$DEST"
PATHS=$(printf '"%s",' "$@")
bash "$EX" --url "$(cat $S/$BOX.url)" --token "$(cat $S/$BOX.token)" - <<PY 2>/dev/null > $S/.pf_$BOX.out
import base64, hashlib, json, os
out = []
for p in [$PATHS]:
    if os.path.exists(p):
        b = open(p, "rb").read(); out.append({"name": os.path.basename(p), "sha": hashlib.sha256(b).hexdigest(), "n": len(b), "b64": base64.b64encode(b).decode()})
print("PULLJSON" + json.dumps(out))
PY
python3 - "$DEST" "$S/.pf_$BOX.out" <<'PY'
import base64, hashlib, json, sys, os
dest, src = sys.argv[1], sys.argv[2]; txt = open(src).read(); i = txt.find("PULLJSON")
if i < 0: print("PULL FAILED:", txt[-300:]); sys.exit(1)
items = json.loads(txt[i+8:].strip().splitlines()[0])
for it in items:
    b = base64.b64decode(it["b64"]); assert hashlib.sha256(b).hexdigest() == it["sha"] and len(b) == it["n"], it["name"]
    open(os.path.join(dest, it["name"]), "wb").write(b)
print("PULL_OK", " ".join(f"{it['name']}({it['n']})" for it in items))
PY

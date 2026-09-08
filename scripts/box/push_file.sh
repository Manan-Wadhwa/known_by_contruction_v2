#!/usr/bin/env bash
# push_file.sh <local_path> <remote_path>
#
# Chunked, offset-addressed, sha256-verified push into box 1 over the marimo
# exec channel. Offset-addressed means a re-run RESUMES: the box appends a
# chunk only when the remote file's current size equals that chunk's offset,
# so an interrupted push is finished, not restarted, and a double-send can
# never duplicate bytes.
#
# Env: PUSH_CHUNK (raw bytes per call, default 4194304), PUSH_MIN_CHUNK
#      (floor when backing off, default 262144), PUSH_RESTART=1 (truncate the
#      remote file first).
set -euo pipefail

SP="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding ${BOX:-box1}.url and ${BOX:-box1}.token}"
EX="$SP/ex.sh"

LOCAL="${1:?usage: push_file.sh <local_path> <remote_path>}"
REMOTE="${2:?usage: push_file.sh <local_path> <remote_path>}"
CHUNK="${PUSH_CHUNK:-4194304}"
MIN_CHUNK="${PUSH_MIN_CHUNK:-262144}"

[ -f "$LOCAL" ] || { echo "PUSH_FAIL no such local file: $LOCAL" >&2; exit 2; }
SIZE=$(wc -c < "$LOCAL")
LSHA=$(sha256sum "$LOCAL" | cut -d' ' -f1)
echo "PUSH  $LOCAL -> $REMOTE"
echo "      bytes=$SIZE local_sha256=$LSHA chunk=$CHUNK"
T0=$(date +%s.%N)

# --- remote state: create the parent dir, report the current size ----------
remote_size() {
  python3 - "$REMOTE" <<'PY' | bash "$EX" - 2>/dev/null | sed -n 's/^RSIZE //p'
import sys
r = sys.argv[1]
print(f'''import os
os.makedirs(os.path.dirname({r!r}), exist_ok=True)
print("RSIZE", os.path.getsize({r!r}) if os.path.exists({r!r}) else 0)''')
PY
}

if [ "${PUSH_RESTART:-0}" = "1" ]; then
  python3 - "$REMOTE" <<'PY' | bash "$EX" - >/dev/null 2>&1
import sys
r = sys.argv[1]
print(f'''import os
os.makedirs(os.path.dirname({r!r}), exist_ok=True)
open({r!r}, "wb").close()
print("TRUNCATED")''')
PY
  echo "      remote truncated (PUSH_RESTART=1)"
fi

OFF="$(remote_size)"
[ -n "$OFF" ] || { echo "PUSH_FAIL could not read remote size" >&2; exit 3; }
if [ "$OFF" -gt "$SIZE" ]; then
  echo "PUSH_FAIL remote is larger ($OFF) than local ($SIZE); rerun with PUSH_RESTART=1" >&2
  exit 4
fi
[ "$OFF" -gt 0 ] && echo "      resuming at offset $OFF"

FAILS=0
while [ "$OFF" -lt "$SIZE" ]; do
  OUT=$(python3 - "$LOCAL" "$REMOTE" "$OFF" "$CHUNK" <<'PY' | bash "$EX" - 2>&1 || true
import base64, sys
lp, rp, off, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(lp, "rb") as fh:
    fh.seek(off)
    data = fh.read(chunk)
b64 = base64.b64encode(data).decode()
print(f'''import base64, os
_B = "{b64}"
_R = {rp!r}
_OFF = {off}
os.makedirs(os.path.dirname(_R), exist_ok=True)
if not os.path.exists(_R):
    open(_R, "wb").close()
cur = os.path.getsize(_R)
if cur != _OFF:
    print("PUSH_MISMATCH", cur)
else:
    blob = base64.b64decode(_B)
    with open(_R, "ab") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())
    print("PUSH_OK", os.path.getsize(_R))''')
PY
)
  NEW=$(printf '%s\n' "$OUT" | sed -n 's/^PUSH_OK //p' | tail -1)
  MIS=$(printf '%s\n' "$OUT" | sed -n 's/^PUSH_MISMATCH //p' | tail -1)
  if [ -n "$NEW" ]; then
    OFF="$NEW"; FAILS=0
    printf '\r      %s / %s bytes (%d%%)   ' "$OFF" "$SIZE" "$((OFF * 100 / SIZE))"
  elif [ -n "$MIS" ]; then
    echo ""; echo "      remote size drifted to $MIS; re-aiming"
    OFF="$MIS"
    [ "$OFF" -gt "$SIZE" ] && { echo "PUSH_FAIL remote overshot" >&2; exit 5; }
  else
    FAILS=$((FAILS + 1))
    echo ""
    echo "      chunk at $OFF failed (attempt $FAILS, chunk=$CHUNK); channel said:" >&2
    printf '%s\n' "$OUT" | tail -3 >&2
    if [ "$CHUNK" -gt "$MIN_CHUNK" ]; then
      CHUNK=$((CHUNK / 2)); echo "      backing off to chunk=$CHUNK" >&2
    fi
    [ "$FAILS" -ge 6 ] && { echo "PUSH_FAIL six consecutive failures at offset $OFF" >&2; exit 6; }
    OFF="$(remote_size)"
  fi
done
echo ""

# --- verify ----------------------------------------------------------------
RES=$(python3 - "$REMOTE" <<'PY' | bash "$EX" - 2>/dev/null
import sys
r = sys.argv[1]
print(f'''import hashlib, os
h = hashlib.sha256()
with open({r!r}, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 20), b""):
        h.update(b)
print("RSHA", h.hexdigest(), os.path.getsize({r!r}))''')
PY
)
RSHA=$(printf '%s\n' "$RES" | sed -n 's/^RSHA \([0-9a-f]*\) .*/\1/p' | tail -1)
RSIZE=$(printf '%s\n' "$RES" | sed -n 's/^RSHA [0-9a-f]* //p' | tail -1)
T1=$(date +%s.%N)
SECS=$(python3 -c "print(f'{$T1-$T0:.1f}')")
MBPS=$(python3 -c "print(f'{$SIZE/1048576/max($T1-$T0,1e-9):.3f}')")

echo "      remote bytes=$RSIZE remote_sha256=$RSHA"
echo "      seconds=$SECS MB/s=$MBPS"
if [ "$RSHA" = "$LSHA" ] && [ "$RSIZE" = "$SIZE" ]; then
  echo "PUSH_VERIFIED $REMOTE $SIZE $LSHA $SECS $MBPS"
  exit 0
fi
echo "PUSH_FAIL sha/size mismatch: local $SIZE/$LSHA remote $RSIZE/$RSHA" >&2
exit 1

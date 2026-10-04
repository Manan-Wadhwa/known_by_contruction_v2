#!/usr/bin/env bash
# rewatch.sh "<JOBS>": restart the durable mz01-watch unit with a new job list (needs MZ01_BOX_DIR).
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
S="${MZ01_BOX_DIR:?set MZ01_BOX_DIR}"
systemctl --user stop mz01-watch 2>/dev/null; sleep 1
systemd-run --user --unit=mz01-watch --collect --setenv=MZ01_BOX_DIR=$S --setenv=JOBS="$1" --setenv=CYCLE=120 --setenv=MAX=3000 --setenv=HOME=/home/manan --setenv=PATH=/usr/local/bin:/usr/bin:/bin --working-directory=$PWD -p StandardOutput=append:$S/watch_all.log -p StandardError=append:$S/watch_all.log bash scripts/box/watch_all.sh
sleep 2; systemctl --user is-active mz01-watch

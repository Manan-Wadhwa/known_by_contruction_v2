#!/usr/bin/env bash
# ex.sh — run python on box1 through the marimo exec channel.
# Usage: ex.sh -c "code"   |   ex.sh -   (code on stdin)
set -euo pipefail
SP="${MZ01_BOX_DIR:?set MZ01_BOX_DIR to the directory holding ${BOX:-box1}.url and ${BOX:-box1}.token}"
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.claude/skills/marimo-pair/scripts/execute-code.sh"
export MARIMO_TOKEN="$(cat "$SP/${BOX:-box1}.token")"
exec bash "$EX" --url "$(cat "$SP/${BOX:-box1}.url")" "$@"

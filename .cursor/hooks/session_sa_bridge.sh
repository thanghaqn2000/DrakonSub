#!/usr/bin/env bash
# Inject SA loop context when a session starts while loop is active.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STATE_FILE="$ROOT/.sa_bridge/state.json"

if [[ ! -f "$STATE_FILE" ]]; then
  exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
  exit 0
fi

loop_active="$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('loop_active', False))" 2>/dev/null || echo false)"
if [[ "$loop_active" != "True" && "$loop_active" != "true" ]]; then
  exit 0
fi

status="$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('status', 'idle'))" 2>/dev/null || echo idle)"
phase="$(python3 -c "import json; s=json.load(open('$STATE_FILE')); print(s.get('phase',''))" 2>/dev/null || echo "")"

cat <<EOF
{
  "additional_context": "SA Bridge loop is ACTIVE (status=${status}, phase=${phase}). Continue SA↔Engineer autoloop per .cursor/rules/sa-engineer-autoloop.mdc. Do not ask the user to wake you or approve steps."
}
EOF

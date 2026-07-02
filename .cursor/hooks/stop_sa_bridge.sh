#!/usr/bin/env bash
# Cursor stop hook: auto-continue SA ↔ Engineer loop without user wake.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STATE_FILE="$ROOT/.sa_bridge/state.json"
CTL="$ROOT/scripts/sa_bridge_ctl.py"
HOOK_PAYLOAD="$(cat 2>/dev/null || true)"

hook_status="completed"
if [[ -n "$HOOK_PAYLOAD" ]] && command -v python3 >/dev/null 2>&1; then
  hook_status="$(HOOK_PAYLOAD="$HOOK_PAYLOAD" python3 - <<'PY'
import json
import os

payload = os.environ.get("HOOK_PAYLOAD", "").strip()
if not payload:
    print("completed")
else:
    try:
        data = json.loads(payload)
    except Exception:
        print("completed")
    else:
        print((data.get("status") or "completed").strip())
PY
)"
fi

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

if [[ "$hook_status" == "aborted" ]]; then
  python3 "$CTL" disarm --reason "Cursor Stop button pressed" >/dev/null
  exit 0
fi

status="$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('status', 'idle'))" 2>/dev/null || echo idle)"
poll_attempts="$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('poll_attempts', 0))" 2>/dev/null || echo 0)"

case "$status" in
  waiting_for_sa)
    python3 "$CTL" bump-poll >/dev/null
    new_status="$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('status', 'idle'))" 2>/dev/null || echo idle)"
    if [[ "$new_status" == "stalled" ]]; then
      exit 0
    fi
    cat <<EOF
{
  "followup_message": "SA Bridge auto-continue (#${poll_attempts}): Poll ChatGPT SA until response ready, then act.\n\n1. Read .cursor/rules/sa-engineer-autoloop.mdc section 'Poll SA'.\n2. browser_tabs → lock ChatGPT tab from .sa_bridge/config.json.\n3. CDP poll: if still streaming, exit turn (hook will re-trigger).\n4. If SA ready: save .sa_bridge/last_sa_response.md, run: python3 scripts/sa_bridge_ctl.py mark-ready\n5. Parse SA task; if DONE for phase → disarm or next phase; else execute task, send ENGINEER REPORT, note-sent, wait again.\n\nDo NOT ask the user. Do NOT stop for approval."
}
EOF
    ;;
  sa_ready)
    cat <<EOF
{
  "followup_message": "SA Bridge auto-continue: SA response is ready.\n\n1. Read .sa_bridge/last_sa_response.md (and state via python3 scripts/sa_bridge_ctl.py status).\n2. python3 scripts/sa_bridge_ctl.py mark-executing --phase \"<current>\"\n3. Execute SA task per .cursor/rules/sa-engineer-autoloop.mdc.\n4. After work: send ENGINEER REPORT to ChatGPT, then note-sent with new assistant count.\n5. If SA said DONE for entire loop: python3 scripts/sa_bridge_ctl.py disarm\n\nDo NOT ask the user."
}
EOF
    ;;
  executing_task)
    cat <<EOF
{
  "followup_message": "SA Bridge auto-continue: finish executing SA task if interrupted, then send ENGINEER REPORT and note-sent. See .cursor/rules/sa-engineer-autoloop.mdc."
}
EOF
    ;;
  *)
    exit 0
    ;;
esac

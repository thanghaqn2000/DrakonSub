#!/usr/bin/env python3
"""File-based state for SA (ChatGPT) ↔ Engineer (Cursor) auto-loop."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / ".sa_bridge"
CONFIG_PATH = BRIDGE / "config.json"
STATE_PATH = BRIDGE / "state.json"
DEFAULT_STATE: Dict[str, Any] = {
    "loop_active": False,
    "status": "idle",
    "phase": "",
    "poll_attempts": 0,
    "assistant_count_at_send": 0,
    "last_message_len_at_send": 0,
    "last_engineer_report_at": None,
    "last_sa_response_at": None,
    "last_error": None,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def load_state() -> dict:
    if not STATE_PATH.exists():
        return dict(DEFAULT_STATE)
    data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    merged = dict(DEFAULT_STATE)
    merged.update(data)
    return merged


def save_state(state: dict) -> None:
    BRIDGE.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_arm(args: argparse.Namespace) -> int:
    state = load_state()
    state["loop_active"] = True
    state["status"] = "idle"
    state["phase"] = args.phase or state.get("phase") or ""
    state["poll_attempts"] = 0
    state["last_error"] = None
    if args.note:
        state["note"] = args.note
    save_state(state)
    print(json.dumps(state, ensure_ascii=False))
    return 0


def cmd_disarm(args: argparse.Namespace) -> int:
    state = load_state()
    state["loop_active"] = False
    state["status"] = "idle"
    state["poll_attempts"] = 0
    if args.reason:
        state["last_error"] = args.reason
    save_state(state)
    print(json.dumps(state, ensure_ascii=False))
    return 0


def cmd_note_sent(args: argparse.Namespace) -> int:
    state = load_state()
    if not state.get("loop_active"):
        state["loop_active"] = True
    state["status"] = "waiting_for_sa"
    state["assistant_count_at_send"] = args.assistant_count
    state["last_message_len_at_send"] = args.message_len
    state["last_engineer_report_at"] = _utc_now()
    state["poll_attempts"] = 0
    state["stable_reads"] = 0
    save_state(state)
    print(json.dumps(state, ensure_ascii=False))
    return 0


def cmd_bump_poll(args: argparse.Namespace) -> int:
    state = load_state()
    state["poll_attempts"] = int(state.get("poll_attempts", 0)) + 1
    cfg = load_config()
    max_attempts = int(cfg.get("max_poll_attempts", 120))
    if state["poll_attempts"] >= max_attempts:
        state["loop_active"] = False
        state["status"] = "stalled"
        state["last_error"] = f"Exceeded max_poll_attempts ({max_attempts})"
    save_state(state)
    print(json.dumps(state, ensure_ascii=False))
    return 0


def cmd_mark_ready(args: argparse.Namespace) -> int:
    state = load_state()
    state["status"] = "sa_ready"
    state["last_sa_response_at"] = _utc_now()
    state["poll_attempts"] = 0
    state["stable_reads"] = 0
    if args.response_file:
        path = Path(args.response_file)
        if path.exists():
            state["last_sa_response_file"] = str(path)
    save_state(state)
    print(json.dumps(state, ensure_ascii=False))
    return 0


def cmd_mark_waiting(args: argparse.Namespace) -> int:
    state = load_state()
    state["status"] = "waiting_for_sa"
    save_state(state)
    print(json.dumps(state, ensure_ascii=False))
    return 0


def cmd_mark_executing(args: argparse.Namespace) -> int:
    state = load_state()
    state["status"] = "executing_task"
    if args.phase:
        state["phase"] = args.phase
    save_state(state)
    print(json.dumps(state, ensure_ascii=False))
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    print(json.dumps({"config": load_config(), "state": load_state()}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SA Bridge state control")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_arm = sub.add_parser("arm", help="Enable auto-loop")
    p_arm.add_argument("--phase", default="")
    p_arm.add_argument("--note", default="")
    p_arm.set_defaults(func=cmd_arm)

    p_disarm = sub.add_parser("disarm", help="Disable auto-loop")
    p_disarm.add_argument("--reason", default="")
    p_disarm.set_defaults(func=cmd_disarm)

    p_sent = sub.add_parser("note-sent", help="Record engineer report sent to SA")
    p_sent.add_argument("--assistant-count", type=int, required=True)
    p_sent.add_argument("--message-len", type=int, default=0)
    p_sent.set_defaults(func=cmd_note_sent)

    sub.add_parser("bump-poll", help="Increment poll counter").set_defaults(func=cmd_bump_poll)

    p_ready = sub.add_parser("mark-ready", help="SA response captured")
    p_ready.add_argument("--response-file", default=str(BRIDGE / "last_sa_response.md"))
    p_ready.set_defaults(func=cmd_mark_ready)

    sub.add_parser("mark-waiting", help="Back to waiting_for_sa").set_defaults(func=cmd_mark_waiting)

    p_exec = sub.add_parser("mark-executing", help="Engineer executing SA task")
    p_exec.add_argument("--phase", default="")
    p_exec.set_defaults(func=cmd_mark_executing)

    sub.add_parser("status", help="Print config + state").set_defaults(func=cmd_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

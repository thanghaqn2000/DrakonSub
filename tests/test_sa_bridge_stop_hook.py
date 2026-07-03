import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / ".sa_bridge" / "state.json"
HOOK_PATH = ROOT / ".cursor" / "hooks" / "stop_sa_bridge.sh"


class SaBridgeStopHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_state = STATE_PATH.read_text(encoding="utf-8") if STATE_PATH.exists() else None
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self._original_state is None:
            if STATE_PATH.exists():
                STATE_PATH.unlink()
        else:
            STATE_PATH.write_text(self._original_state, encoding="utf-8")

    def _write_state(self, **overrides) -> None:
        state = {
            "loop_active": True,
            "status": "waiting_for_sa",
            "phase": "test-phase",
            "poll_attempts": 0,
            "assistant_count_at_send": 1,
            "last_message_len_at_send": 10,
            "last_engineer_report_at": None,
            "last_sa_response_at": None,
            "last_error": None,
        }
        state.update(overrides)
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _run_hook(self, payload: dict) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(HOOK_PATH)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            cwd=str(ROOT),
            check=False,
        )

    def test_aborted_stop_disarms_and_emits_no_followup(self) -> None:
        self._write_state(loop_active=True, status="waiting_for_sa")

        result = self._run_hook({"status": "aborted", "loop_count": 0})

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        self.assertFalse(state["loop_active"])
        self.assertEqual(state["status"], "idle")

    def test_completed_stop_keeps_autocontinue_flow(self) -> None:
        self._write_state(loop_active=True, status="waiting_for_sa", poll_attempts=2)

        result = self._run_hook({"status": "completed", "loop_count": 0})

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("followup_message", result.stdout)
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        self.assertTrue(state["loop_active"])
        self.assertEqual(state["status"], "waiting_for_sa")
        self.assertEqual(state["poll_attempts"], 3)


if __name__ == "__main__":
    unittest.main()

# SRT → Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new **SRT → Audio** tab that uploads an SRT, lets users edit text/timestamps, validates length via estimate (no auto speed/shift), synthesizes Saydi TTS onto a silence timeline, and lets users play/download WAV or MP3.

**Architecture:** New `srt_audio` package + `/api/srt-audio/*` job APIs (separate from video voiceover). Reuse `subtitle_edit_service` for SRT I/O, `saydi_tts` for synthesis, and an ffmpeg `adelay` mixer similar to `build_voiceover_track` without video mux. Pre-check blocks synthesize when estimated speech > cue slot or cues overlap.

**Tech Stack:** Python FastAPI, existing Saydi HTTP TTS, ffmpeg, unittest, `index.html` mode tabs.

**Spec:** `docs/superpowers/specs/2026-07-14-srt-to-audio-design.md`  
**Branch:** `feature/implement_srt_feature`

**Locked details:**
- Estimate: `estimated_ms = ceil((char_count / chars_per_second) * 1000 / max(saydi_speed, 0.01))` with default `chars_per_second=13.0` (env `SRT_AUDIO_CHARS_PER_SECOND`).
- No separate `/validate` endpoint; `GET/PUT .../cues` returns per-cue `issues` + `estimated_ms`.
- Synthesize failure: clear error on job; keep partial segments optional but status=`failed` with message (no auto-retry).
- MVP: trust pre-check; after TTS only log overflow warning in manifest, never shift/atruncate.

---

## File map

| File | Responsibility |
|------|----------------|
| `auto_subtitle/srt_audio/__init__.py` | Package marker |
| `auto_subtitle/srt_audio/timing.py` | Parse timestamps, overlap checks, estimate |
| `auto_subtitle/srt_audio/cue_service.py` | Load/save cues, annotate issues |
| `auto_subtitle/srt_audio/job_service.py` | Create job, synthesize async pipeline, convert MP3 |
| `auto_subtitle/srt_audio/audio_track.py` | Build mono WAV timeline via ffmpeg adelay |
| `auto_subtitle/web.py` | HTTP routes under `/api/srt-audio` |
| `auto_subtitle/static/index.html` | New mode tab + UI steps |
| `.env.example` | `DRAKONSUB_SRT_AUDIO_JOBS_ROOT`, `SRT_AUDIO_CHARS_PER_SECOND` |
| `tests/test_srt_audio_timing.py` | Estimate + validation |
| `tests/test_srt_audio_job_service.py` | Job create / synthesize (mock Saydi) |
| `tests/test_srt_audio_web.py` | API smoke (TestClient) |

---

### Task 1: Timing helpers (estimate + hard validation)

**Files:**
- Create: `auto_subtitle/srt_audio/__init__.py`
- Create: `auto_subtitle/srt_audio/timing.py`
- Test: `tests/test_srt_audio_timing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_srt_audio_timing.py
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.srt_audio.timing import (  # noqa: E402
    estimate_speech_ms,
    ms_to_srt_timestamp,
    parse_srt_timestamp_to_ms,
    validate_cue_timings,
)


class SrtAudioTimingTests(unittest.TestCase):
    def test_parse_and_format_roundtrip(self) -> None:
        ms = parse_srt_timestamp_to_ms("00:01:02,345")
        self.assertEqual(ms, 62_345)
        self.assertEqual(ms_to_srt_timestamp(ms), "00:01:02,345")

    def test_estimate_scales_with_speed(self) -> None:
        base = estimate_speech_ms("abcdefghij", chars_per_second=10.0, saydi_speed=1.0)
        fast = estimate_speech_ms("abcdefghij", chars_per_second=10.0, saydi_speed=2.0)
        self.assertEqual(base, 1_000)
        self.assertEqual(fast, 500)

    def test_validate_flags_overlap_empty_and_too_long(self) -> None:
        cues = [
            {"index": 1, "start": "00:00:00,000", "end": "00:00:01,000", "text": "A" * 40},
            {"index": 2, "start": "00:00:00,500", "end": "00:00:02,000", "text": "ok"},
            {"index": 3, "start": "00:00:03,000", "end": "00:00:02,000", "text": "  "},
        ]
        issues = validate_cue_timings(cues, chars_per_second=13.0, saydi_speed=1.0)
        self.assertIn("overlap_next", issues[0])
        self.assertIn("too_long", issues[0])
        self.assertIn("start_after_end", issues[2])
        self.assertIn("empty_text", issues[2])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests — expect import/fail**

```bash
python3 -m unittest tests.test_srt_audio_timing -v
```

Expected: FAIL (`ModuleNotFoundError` or import error).

- [ ] **Step 3: Implement timing module**

```python
# auto_subtitle/srt_audio/__init__.py
"""SRT → Audio narration jobs (no video)."""

# auto_subtitle/srt_audio/timing.py
from __future__ import annotations

import math
from typing import Any


def parse_srt_timestamp_to_ms(timestamp: str) -> int:
    hours, minutes, rest = timestamp.strip().split(":")
    seconds, millis = rest.split(",")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(millis)
    )


def ms_to_srt_timestamp(value: int) -> str:
    value = max(0, int(value))
    hours = value // 3_600_000
    value %= 3_600_000
    minutes = value // 60_000
    value %= 60_000
    seconds = value // 1_000
    millis = value % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def estimate_speech_ms(
    text: str,
    *,
    chars_per_second: float = 13.0,
    saydi_speed: float = 1.0,
) -> int:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return 0
    cps = max(float(chars_per_second), 0.1)
    speed = max(float(saydi_speed), 0.01)
    return int(math.ceil((len(cleaned) / cps) * 1000.0 / speed))


def validate_cue_timings(
    cues: list[dict[str, Any]],
    *,
    chars_per_second: float = 13.0,
    saydi_speed: float = 1.0,
) -> list[list[str]]:
    """Return per-cue issue codes: empty_text, start_after_end, overlap_next, too_long."""
    result: list[list[str]] = [[] for _ in cues]
    starts: list[int] = []
    ends: list[int] = []
    for idx, cue in enumerate(cues):
        text = str(cue.get("text") or "").strip()
        if not text:
            result[idx].append("empty_text")
        try:
            start_ms = parse_srt_timestamp_to_ms(str(cue.get("start") or ""))
            end_ms = parse_srt_timestamp_to_ms(str(cue.get("end") or ""))
        except (ValueError, AttributeError):
            result[idx].append("invalid_timestamp")
            starts.append(0)
            ends.append(0)
            continue
        starts.append(start_ms)
        ends.append(end_ms)
        if start_ms >= end_ms:
            result[idx].append("start_after_end")
        duration = max(0, end_ms - start_ms)
        estimated = estimate_speech_ms(
            text, chars_per_second=chars_per_second, saydi_speed=saydi_speed
        )
        if duration > 0 and estimated > duration:
            result[idx].append("too_long")

    for idx in range(len(cues) - 1):
        if "invalid_timestamp" in result[idx] or "invalid_timestamp" in result[idx + 1]:
            continue
        if "start_after_end" in result[idx]:
            continue
        if ends[idx] > starts[idx + 1]:
            result[idx].append("overlap_next")
    return result
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python3 -m unittest tests.test_srt_audio_timing -v
```

- [ ] **Step 5: Commit**

```bash
git add auto_subtitle/srt_audio/__init__.py auto_subtitle/srt_audio/timing.py tests/test_srt_audio_timing.py
git commit -m "Add SRT-audio timing estimate and cue validation."
```

---

### Task 2: Cue service (annotate issues for API)

**Files:**
- Create: `auto_subtitle/srt_audio/cue_service.py`
- Test: `tests/test_srt_audio_cue_service.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_srt_audio_cue_service.py
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.srt_audio.cue_service import (  # noqa: E402
    annotate_cues,
    save_edited_cues,
    load_effective_cues,
)
from auto_subtitle.subtitle_edit_service import SubtitleCue, write_srt  # noqa: E402


class SrtAudioCueServiceTests(unittest.TestCase):
    def test_annotate_includes_estimated_ms_and_issues(self) -> None:
        cues = [
            SubtitleCue(1, "00:00:00,000", "00:00:01,000", "short"),
            SubtitleCue(2, "00:00:01,000", "00:00:02,000", "x" * 80),
        ]
        rows = annotate_cues(cues, chars_per_second=13.0, saydi_speed=1.0)
        self.assertEqual(rows[0]["issues"], [])
        self.assertIn("too_long", rows[1]["issues"])
        self.assertGreater(rows[1]["estimated_ms"], 1000)

    def test_save_rejects_count_mismatch_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmpdir := tmp)
            original = [
                SubtitleCue(1, "00:00:00,000", "00:00:01,000", "a"),
                SubtitleCue(2, "00:00:01,000", "00:00:02,000", "b"),
            ]
            write_srt(original, job_dir / "input.srt")
            with self.assertRaises(ValueError):
                save_edited_cues(job_dir, [{"index": 1, "start": "00:00:00,000", "end": "00:00:01,000", "text": "a"}])
            save_edited_cues(
                job_dir,
                [
                    {"index": 1, "start": "00:00:00,000", "end": "00:00:00,800", "text": "A"},
                    {"index": 2, "start": "00:00:01,000", "end": "00:00:02,000", "text": "B"},
                ],
            )
            loaded = load_effective_cues(job_dir)
            self.assertEqual(loaded[0].text, "A")
            self.assertTrue((job_dir / "edited.srt").is_file())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python3 -m unittest tests.test_srt_audio_cue_service -v
```

- [ ] **Step 3: Implement cue_service**

```python
# auto_subtitle/srt_audio/cue_service.py
from __future__ import annotations

from pathlib import Path

from auto_subtitle.subtitle_edit_service import SubtitleCue, SubtitleEditError, load_srt, write_srt

from .timing import estimate_speech_ms, validate_cue_timings


class SrtAudioCueError(ValueError):
    pass


def input_srt_path(job_dir: Path) -> Path:
    return job_dir / "input.srt"


def edited_srt_path(job_dir: Path) -> Path:
    return job_dir / "edited.srt"


def load_effective_cues(job_dir: Path) -> list[SubtitleCue]:
    edited = edited_srt_path(job_dir)
    if edited.is_file():
        return load_srt(edited)
    return load_srt(input_srt_path(job_dir))


def annotate_cues(
    cues: list[SubtitleCue],
    *,
    chars_per_second: float = 13.0,
    saydi_speed: float = 1.0,
) -> list[dict]:
    payload = [
        {"index": c.index, "start": c.start, "end": c.end, "text": c.text}
        for c in cues
    ]
    issues = validate_cue_timings(
        payload, chars_per_second=chars_per_second, saydi_speed=saydi_speed
    )
    rows: list[dict] = []
    for cue, cue_issues in zip(cues, issues):
        estimated = estimate_speech_ms(
            cue.text, chars_per_second=chars_per_second, saydi_speed=saydi_speed
        )
        rows.append(
            {
                "index": cue.index,
                "start": cue.start,
                "end": cue.end,
                "text": cue.text,
                "estimated_ms": estimated,
                "issues": cue_issues,
            }
        )
    return rows


def save_edited_cues(job_dir: Path, submitted: list[dict]) -> list[SubtitleCue]:
    original = load_srt(input_srt_path(job_dir))
    if len(submitted) != len(original):
        raise SrtAudioCueError("Số lượng cue không khớp.")
    by_index = {c.index: c for c in original}
    updated: list[SubtitleCue] = []
    seen: set[int] = set()
    for item in submitted:
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError) as exc:
            raise SrtAudioCueError("Index cue không hợp lệ.") from exc
        if index in seen or index not in by_index:
            raise SrtAudioCueError("Index cue không khớp.")
        seen.add(index)
        start = str(item.get("start", "")).strip()
        end = str(item.get("end", "")).strip()
        text = str(item.get("text", "")).strip()
        if not text:
            raise SrtAudioCueError("Nội dung cue không được để trống.")
        updated.append(SubtitleCue(index=index, start=start, end=end, text=text))
    updated.sort(key=lambda c: c.index)
    if [c.index for c in updated] != [c.index for c in original]:
        raise SrtAudioCueError("Index cue không khớp.")
    write_srt(updated, edited_srt_path(job_dir))
    return updated


def has_blocking_issues(rows: list[dict]) -> bool:
    return any(row.get("issues") for row in rows)
```

- [ ] **Step 4: Run — expect PASS**

```bash
python3 -m unittest tests.test_srt_audio_cue_service -v
```

- [ ] **Step 5: Commit**

```bash
git add auto_subtitle/srt_audio/cue_service.py tests/test_srt_audio_cue_service.py
git commit -m "Add SRT-audio cue load/save with issue annotations."
```

---

### Task 3: Audio track builder + job synthesize (mock Saydi)

**Files:**
- Create: `auto_subtitle/srt_audio/audio_track.py`
- Create: `auto_subtitle/srt_audio/job_service.py`
- Test: `tests/test_srt_audio_job_service.py`

- [ ] **Step 1: Write failing tests for job create + synthesize reject + happy path**

```python
# tests/test_srt_audio_job_service.py
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.srt_audio.job_service import (  # noqa: E402
    SrtAudioJobError,
    create_job_from_srt_bytes,
    run_synthesize_job,
)


class SrtAudioJobServiceTests(unittest.TestCase):
    def test_create_job_writes_input_and_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            srt = b"1\n00:00:00,000 --> 00:00:01,000\nXin chao\n"
            job_id, job_dir = create_job_from_srt_bytes(root, srt)
            self.assertTrue((job_dir / "input.srt").is_file())
            meta = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["status"], "ready")
            self.assertEqual(meta["cue_count"], 1)
            self.assertEqual(job_id, job_dir.name)

    def test_synthesize_rejects_too_long_before_saydi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            srt = (
                "1\n00:00:00,000 --> 00:00:01,000\n"
                + ("dai " * 40)
                + "\n"
            ).encode("utf-8")
            _, job_dir = create_job_from_srt_bytes(root, srt)
            with patch("auto_subtitle.srt_audio.job_service.synthesize_to_file") as mock_tts:
                with self.assertRaises(SrtAudioJobError):
                    run_synthesize_job(
                        job_dir,
                        saydi_sample=None,
                        saydi_speed=1.0,
                        output_format="wav",
                        chars_per_second=13.0,
                    )
                mock_tts.assert_not_called()

    @patch("auto_subtitle.srt_audio.job_service.convert_wav_to_mp3")
    @patch("auto_subtitle.srt_audio.job_service.build_srt_audio_track")
    @patch("auto_subtitle.srt_audio.job_service.probe_audio_duration_ms", return_value=800)
    @patch("auto_subtitle.srt_audio.job_service.synthesize_to_file")
    @patch(
        "auto_subtitle.srt_audio.job_service.load_saydi_config",
        return_value=type(
            "Cfg",
            (),
            {"token": "x", "sample": "s", "speed": 1.0, "lang": "vi", "output_format": "wav"},
        )(),
    )
    def test_synthesize_builds_wav(
        self, _cfg, _tts, _probe, mock_build, mock_mp3
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            srt = b"1\n00:00:00,000 --> 00:00:02,000\nXin chao\n"
            _, job_dir = create_job_from_srt_bytes(root, srt)

            def _fake_build(*, segment_starts_ms, segment_paths, track_duration_ms, output_path):
                Path(output_path).write_bytes(b"RIFF")

            mock_build.side_effect = _fake_build
            result = run_synthesize_job(
                job_dir,
                saydi_sample=None,
                saydi_speed=1.0,
                output_format="wav",
                chars_per_second=13.0,
            )
            self.assertTrue(Path(result["output_wav"]).is_file())
            mock_mp3.assert_not_called()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python3 -m unittest tests.test_srt_audio_job_service -v
```

- [ ] **Step 3: Implement `audio_track.py` and `job_service.py`**

`audio_track.py` — copy the adelay approach from `auto_subtitle/voiceover/audio_builder.py::build_voiceover_track`, but take `track_duration_ms` instead of `video_duration_ms`, and inputs as parallel lists `(segment_path, start_ms)`.

`job_service.py` — key functions:

```python
def create_job_from_srt_bytes(jobs_root: Path, srt_bytes: bytes) -> tuple[str, Path]:
    # uuid job dir, write input.srt, parse via load_srt, write job.json status=ready

def run_synthesize_job(job_dir, *, saydi_sample, saydi_speed, output_format, chars_per_second) -> dict:
    # load effective cues → annotate → if has_blocking_issues: raise SrtAudioJobError
    # load_saydi_config; require token
    # for each cue: synthesize_to_file → probe duration; optionally append overflow warning
    # track_duration_ms = max(last_end_ms, max(start+tts for each))
    # build_srt_audio_track(...) → output.wav
    # if output_format == "mp3": convert_wav_to_mp3 → output.mp3
    # write manifest.json + update job.json status=completed
```

`convert_wav_to_mp3`:

```python
subprocess.run(
    ["ffmpeg", "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-q:a", "2", str(mp3)],
    check=False, capture_output=True, text=True,
)
```

- [ ] **Step 4: Run — expect PASS**

```bash
python3 -m unittest tests.test_srt_audio_job_service -v
```

- [ ] **Step 5: Commit**

```bash
git add auto_subtitle/srt_audio/audio_track.py auto_subtitle/srt_audio/job_service.py tests/test_srt_audio_job_service.py
git commit -m "Add SRT-audio synthesize job and timeline track builder."
```

---

### Task 4: FastAPI routes `/api/srt-audio`

**Files:**
- Modify: `auto_subtitle/web.py`
- Modify: `.env.example`
- Test: `tests/test_srt_audio_web.py`

- [ ] **Step 1: Write API tests with FastAPI TestClient** (pattern from `tests/test_voiceover_web.py`)

Cover:
1. `POST /api/srt-audio/jobs` with SRT upload → `job_id`, status ready
2. `GET /api/srt-audio/jobs/{id}/cues` → annotated cues
3. `PUT` cues updates text/timestamps
4. `POST .../synthesize` with mocked `run_synthesize_job` → status synthesizing then completed via background (or sync-invoke in test by calling target directly / patching Thread)
5. `GET .../audio?format=wav` when completed

Prefer patching `threading.Thread` to run target synchronously in tests.

- [ ] **Step 2: Run — expect FAIL**

```bash
python3 -m unittest tests.test_srt_audio_web -v
```

- [ ] **Step 3: Wire routes in `web.py`**

Add:

```python
def _resolve_srt_audio_jobs_root() -> Path:
    raw = os.getenv("DRAKONSUB_SRT_AUDIO_JOBS_ROOT", "").strip()
    if raw:
        return Path(raw)
    return Path("data/srt-audio-jobs")

SRT_AUDIO_JOBS_ROOT = _resolve_srt_audio_jobs_root()
```

Endpoints:
- `POST /api/srt-audio/jobs` — `UploadFile` `.srt`
- `GET /api/srt-audio/jobs/{job_id}`
- `GET /api/srt-audio/jobs/{job_id}/cues?saydi_speed=&chars_per_second=`
- `PUT /api/srt-audio/jobs/{job_id}/cues` — body `{cues, saydi_speed?}`
- `POST /api/srt-audio/jobs/{job_id}/synthesize` — body `{saydi_sample, saydi_speed, output_format}`
- `GET /api/srt-audio/jobs/{job_id}/audio?format=wav|mp3`

On synthesize: set status `processing`, spawn thread calling `run_synthesize_job`, on success `completed`, on error `failed` + message.

Also expose saydi presets via existing `/api/voiceover/config` or reuse same helper in UI.

`.env.example` additions:

```
DRAKONSUB_SRT_AUDIO_JOBS_ROOT=data/srt-audio-jobs
SRT_AUDIO_CHARS_PER_SECOND=13
```

docker-compose: volume mount `./data/srt-audio-jobs:/app/data/srt-audio-jobs` and env `DRAKONSUB_SRT_AUDIO_JOBS_ROOT=/app/data/srt-audio-jobs`.

- [ ] **Step 4: Tests PASS**

```bash
python3 -m unittest tests.test_srt_audio_web -v
```

- [ ] **Step 5: Commit**

```bash
git add auto_subtitle/web.py .env.example docker-compose.yml tests/test_srt_audio_web.py
git commit -m "Expose SRT-audio job APIs for upload, edit, and synthesize."
```

---

### Task 5: UI tab **SRT → Audio**

**Files:**
- Modify: `auto_subtitle/static/index.html`

- [ ] **Step 1: Add mode tab + panel HTML**

Next to existing tabs:

```html
<button class="mode-tab" id="mode-srt-audio-tab" type="button">SRT → Audio</button>
```

Panel sections:
1. Upload SRT
2. Cue editor (reuse voiceover cue editor styling; editable start/end/text)
3. Saydi sample select, speed, format select (wav|mp3), button **Thuyết minh** (disabled if any `issues`)
4. Status + `<audio controls id="srt-audio-player">` + download link

- [ ] **Step 2: Extend `setMode` for three modes**

```javascript
function setMode(mode) {
  currentMode = mode;
  modeVietsubTab.classList.toggle("active", mode === "vietsub");
  modeVoiceoverTab.classList.toggle("active", mode === "voiceover");
  modeSrtAudioTab.classList.toggle("active", mode === "srt-audio");
  vietsubPanel.classList.toggle("active", mode === "vietsub");
  voiceoverPanel.classList.toggle("active", mode === "voiceover");
  srtAudioPanel.classList.toggle("active", mode === "srt-audio");
}
```

- [ ] **Step 3: Wire JS flow**

- Upload → `POST /api/srt-audio/jobs` FormData
- Load cues → render rows; on input recompute local disable of synthesize if any visible issues from last server annotate (after save or on load). Call GET cues with current speed when speed changes.
- Save → `PUT .../cues`
- Synthesize → `POST .../synthesize` then poll `GET .../jobs/{id}` until `completed|failed`
- On completed: set `audio.src = /api/srt-audio/jobs/{id}/audio?format=...` and show download `<a download>`

Vietnamese copy for errors (empty SRT, validation failed, Saydi missing token).

- [ ] **Step 4: Manual UI check** (local/docker)

Upload `sample-srt.srt` (or a short 2-cue SRT), edit one timestamp, confirm button disables when text too long for slot, then fix and synthesize with real Saydi token if available.

- [ ] **Step 5: Commit**

```bash
git add auto_subtitle/static/index.html
git commit -m "Add SRT → Audio UI tab for edit and download."
```

---

### Task 6: Final verification + PR

- [ ] **Step 1: Run full related tests**

```bash
python3 -m unittest \
  tests.test_srt_audio_timing \
  tests.test_srt_audio_cue_service \
  tests.test_srt_audio_job_service \
  tests.test_srt_audio_web \
  -v
```

Expected: all OK.

- [ ] **Step 2: Rebuild docker and smoke on LAN**

```bash
docker compose up -d --build drakonsub
curl -s http://127.0.0.1:8000/api/health
```

Open `http://192.168.1.5:8000` → tab SRT → Audio.

- [ ] **Step 3: Push + open PR**

```bash
git push -u origin HEAD
gh pr create --title "Add SRT → Audio narration from uploaded SRT" --body "$(cat <<'EOF'
## Summary
- New **SRT → Audio** tab: upload SRT, edit text/timestamps, synthesize Saydi timeline audio
- Pre-check blocks generate when estimated speech exceeds cue slot or timings overlap
- Play + download WAV or MP3

## Test plan
- [ ] Unit tests above pass
- [ ] Upload short SRT, edit cue, synthesize, play/download WAV
- [ ] Confirm too-long cue disables synthesize
- [ ] Download MP3 works
- [ ] Existing Thuyết minh (video) tab unchanged

EOF
)"
```

---

## Spec coverage check

| Spec item | Task |
|-----------|------|
| Tab mới | Task 5 |
| Upload SRT + editor text/timestamp only | Task 2, 4, 5 |
| Estimate pre-check, no auto speed/shift | Task 1, 3 |
| Saydi synthesize + silence timeline | Task 3 |
| Play + download WAV/MP3 | Task 3, 4, 5 |
| Separate from video voiceover | Task 4 file roots + APIs |
| Tests | Tasks 1–4, 6 |

## Placeholder scan

No TBD/TODO steps; estimate formula and storage paths locked above.

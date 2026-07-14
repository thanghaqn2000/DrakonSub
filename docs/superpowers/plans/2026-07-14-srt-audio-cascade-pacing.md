# SRT → Audio Cascade Pacing + Editor UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Place SRT→Audio segments with measured-duration cascade (gap 280ms), persist updated cue timestamps after synthesize, keep user-chosen Saydi speed (no auto fill), and modernize the cue editor with auto-growing textareas.

**Architecture:** Pure helper `plan_cascade_starts` in `timing.py` decides each `planned_start_ms`. `run_synthesize_job` TTS → probe → cascade → `build_srt_audio_track`, then `write_srt` overwrites `edited.srt` with cascade start/end. UI refreshes cues on job complete and renders card-style rows with auto-resize textareas.

**Tech Stack:** Python unittest, existing Saydi mocks, FastAPI job APIs, `index.html` CSS/JS.

**Spec:** `docs/superpowers/specs/2026-07-14-srt-audio-cascade-pacing-design.md`  
**Branch:** `feature/implement_srt_feature`

**Locked details:**
- `GAP_MS` default **280**; env `SRT_AUDIO_CUE_GAP_MS`.
- `planned_start = max(intent_start, prev_end + gap)` for every cue after the first floor (`prev_end` before first cue = `-gap` so first cue uses `max(intent, 0)` effectively: use `prev_end_ms = -gap_ms` initially OR document `prev_end = None` and only apply floor from cue 2+ / after first).
  - **Concrete rule:** initialize `prev_end_ms = None`. Cue 0: `planned_start = max(0, intent_start)`. After each cue: `prev_end = planned_start + duration`. Cue n>0: `planned_start = max(intent_start, prev_end + gap_ms)`.
- Cue `end` after synth = `planned_start + duration_ms` (SRT timestamp).
- No per-cue speed change.
- `too_long` remains warning-only (already implemented).

---

## File map

| File | Responsibility |
|------|----------------|
| `auto_subtitle/srt_audio/timing.py` | Add `plan_cascade_starts(...)` |
| `auto_subtitle/srt_audio/job_service.py` | Cascade loop; write `edited.srt`; rich manifest segments |
| `auto_subtitle/config.py` | Optional constant `SRT_AUDIO_CUE_GAP_MS = 280` (or read env only in web/job) |
| `.env.example` | Document `SRT_AUDIO_CUE_GAP_MS` |
| `auto_subtitle/static/index.html` | Card editor + auto-grow; refresh cues on complete; optional help copy |
| `tests/test_srt_audio_timing.py` | Cascade helper unit tests |
| `tests/test_srt_audio_job_service.py` | Mock TTS durations → assert starts + edited.srt |

---

### Task 1: Cascade placement helper

**Files:**
- Modify: `auto_subtitle/srt_audio/timing.py`
- Modify: `tests/test_srt_audio_timing.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_srt_audio_timing.py`:

```python
from auto_subtitle.srt_audio.timing import plan_cascade_starts


class CascadePlacementTests(unittest.TestCase):
    def test_respects_intent_when_gap_available(self) -> None:
        # cue0 ends 3000; cue1 intent 4000; gap 280 → 3000+280=3280 < 4000 → keep 4000
        starts = plan_cascade_starts(
            intent_starts_ms=[0, 4000],
            durations_ms=[3000, 1000],
            gap_ms=280,
        )
        self.assertEqual(starts, [0, 4000])

    def test_pushes_next_when_overflow(self) -> None:
        # cue0: start 0, dur 5000 → end 5000; cue1 intent 3000 → push to 5280
        starts = plan_cascade_starts(
            intent_starts_ms=[0, 3000],
            durations_ms=[5000, 2000],
            gap_ms=280,
        )
        self.assertEqual(starts, [0, 5280])

    def test_cascade_chain(self) -> None:
        starts = plan_cascade_starts(
            intent_starts_ms=[3000, 11000, 17000],
            durations_ms=[9000, 7000, 1000],
            gap_ms=280,
        )
        # 0: 3000..12000; 1: max(11000, 12280)=12280..19280; 2: max(17000, 19560)=19560
        self.assertEqual(starts, [3000, 12280, 19560])

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            plan_cascade_starts([0], [1, 2], gap_ms=280)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_srt_audio_timing.CascadePlacementTests -v`  
Expected: FAIL (`ImportError` / missing `plan_cascade_starts`)

- [ ] **Step 3: Implement helper**

Add to `auto_subtitle/srt_audio/timing.py`:

```python
def plan_cascade_starts(
    intent_starts_ms: list[int],
    durations_ms: list[int],
    *,
    gap_ms: int = 280,
) -> list[int]:
    if len(intent_starts_ms) != len(durations_ms):
        raise ValueError("intent_starts_ms and durations_ms length mismatch")
    if gap_ms < 0:
        raise ValueError("gap_ms must be >= 0")
    planned: list[int] = []
    prev_end: int | None = None
    for intent, duration in zip(intent_starts_ms, durations_ms):
        intent_i = max(0, int(intent))
        dur_i = max(0, int(duration))
        if prev_end is None:
            start = intent_i
        else:
            start = max(intent_i, prev_end + int(gap_ms))
        planned.append(start)
        prev_end = start + dur_i
    return planned
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `python -m unittest tests.test_srt_audio_timing -v`

- [ ] **Step 5: Commit**

```bash
git add auto_subtitle/srt_audio/timing.py tests/test_srt_audio_timing.py
git commit -m "feat(srt-audio): add cascade start placement helper"
```

---

### Task 2: Wire cascade into synthesize + persist cues

**Files:**
- Modify: `auto_subtitle/srt_audio/job_service.py`
- Modify: `tests/test_srt_audio_job_service.py`
- Modify: `.env.example` (document `SRT_AUDIO_CUE_GAP_MS=280`)
- Optionally read gap in `run_synthesize_job` via `os.getenv("SRT_AUDIO_CUE_GAP_MS", "280")` or kwarg `cue_gap_ms: int = 280`

- [ ] **Step 1: Write failing tests**

In `tests/test_srt_audio_job_service.py`, add a test that mocks two cues overlapping in speech:

```python
def test_synthesize_cascades_and_rewrites_edited_srt(self) -> None:
    sample = (
        "1\n00:00:00,000 --> 00:00:02,000\nOne.\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nTwo.\n\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        job_id, job_dir = create_job_from_srt_bytes(root, sample.encode("utf-8"))
        # preload edited same as input
        from auto_subtitle.srt_audio.cue_service import load_effective_cues, edited_srt_path
        from auto_subtitle.subtitle_edit_service import write_srt, load_srt

        durations = {1: 3500, 2: 1000}  # cue1 speech past cue2 intent

        def fake_tts(text, path, config=None):
            path.write_bytes(b"RIFF....")

        def fake_probe(path):
            # path name 0001.wav / 0002.wav
            idx = int(path.stem)
            return durations[idx]

        starts_captured = {}

        def fake_build(*, segment_starts_ms, segment_paths, track_duration_ms, output_path):
            starts_captured["starts"] = list(segment_starts_ms)
            starts_captured["track"] = track_duration_ms
            output_path.write_bytes(b"wav")

        with patch("auto_subtitle.srt_audio.job_service.load_saydi_config") as cfg:
            cfg.return_value = type("C", (), {"token": "t", "sample": "s", "speed": 1.0})()
            with patch("auto_subtitle.srt_audio.job_service.synthesize_to_file", side_effect=fake_tts):
                with patch("auto_subtitle.srt_audio.job_service.probe_audio_duration_ms", side_effect=fake_probe):
                    with patch("auto_subtitle.srt_audio.job_service.build_srt_audio_track", side_effect=fake_build):
                        run_synthesize_job(
                            job_dir,
                            saydi_sample="s",
                            saydi_speed=1.0,
                            output_format="wav",
                            cue_gap_ms=280,
                        )

        self.assertEqual(starts_captured["starts"], [0, 3780])  # 3500+280
        cues = load_srt(edited_srt_path(job_dir))
        self.assertEqual(cues[0].start, "00:00:00,000")
        self.assertEqual(cues[0].end, "00:00:03,500")
        self.assertEqual(cues[1].start, "00:00:03,780")
        self.assertEqual(cues[1].end, "00:00:04,780")
        manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["cue_gap_ms"], 280)
        self.assertEqual(manifest["segments"][1]["shift_ms"], 1780)
        self.assertEqual(manifest["saydi_speed"], 1.0)
```

Also update any existing test that asserts `segment_starts_ms` equals raw SRT starts if durations would now cascade — fix expected values.

- [ ] **Step 2: Run test — expect FAIL** (still uses raw starts / no edited rewrite)

Run: `python -m unittest tests.FAKESECRET_a1b2c3d4e5f6g7h8i9j0 -v`

- [ ] **Step 3: Implement in `run_synthesize_job`**

Replace the TTS loop body with:

1. Collect `intent_starts`, synthesize all segments, probe `durations`.
2. `planned = plan_cascade_starts(intent_starts, durations, gap_ms=cue_gap_ms)`.
3. Build updated `SubtitleCue` list: `start=ms_to_srt_timestamp(planned[i])`, `end=ms_to_srt_timestamp(planned[i]+durations[i])`, same text/index.
4. `write_srt(updated, edited_srt_path(job_dir))`.
5. `build_srt_audio_track(segment_starts_ms=planned, ...)`.
6. Manifest:

```python
"cue_gap_ms": cue_gap_ms,
"segments": [
  {
    "index": cue.index,
    "text": cue.text,
    "intent_start_ms": intent,
    "planned_start_ms": planned_i,
    "duration_ms": dur,
    "planned_end_ms": planned_i + dur,
    "shift_ms": planned_i - intent,
  },
  ...
],
"cues": annotate_cues(updated_cues, ...),  # post-cascade
```

Add kwarg `cue_gap_ms: int | None = None` and resolve:

```python
import os
...
if cue_gap_ms is None:
    cue_gap_ms = int((os.getenv("SRT_AUDIO_CUE_GAP_MS") or "280").strip() or "280")
```

Keep `overflow_warnings` optional (e.g. when `shift_ms > 0`) or deprecate old `tts_longer_than_slot` in favor of `shift_ms` in segments — prefer logging segments only; may keep warnings list for `shift_ms > 0` as soft signal in `summary`.

Do **not** change `saydi_config.speed` per cue.

- [ ] **Step 4: Run job_service + timing tests — PASS**

Run: `python -m unittest tests.test_srt_audio_job_service tests.test_srt_audio_timing -v`

- [ ] **Step 5: Commit**

```bash
git add auto_subtitle/srt_audio/job_service.py tests/test_srt_audio_job_service.py .env.example
git commit -m "feat(srt-audio): cascade place TTS and rewrite cue timestamps"
```

---

### Task 3: UI — refresh timestamps after synth + editor cards

**Files:**
- Modify: `auto_subtitle/static/index.html`

- [ ] **Step 1: CSS for cue cards**

Near other SRT styles (or in `<style>`), add:

```css
.srt-audio-cue-list { display: flex; flex-direction: column; gap: 0.75rem; }
.srt-audio-cue-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  padding: 0.85rem 1rem;
}
.srt-audio-cue-card.has-error { border-color: #c0392b; }
.srt-audio-cue-card.has-warning { border-color: #d68910; }
.srt-audio-cue-head {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 0.75rem;
  margin-bottom: 0.55rem;
}
.srt-audio-cue-index {
  font-weight: 650; letter-spacing: 0.02em;
  min-width: 2rem;
}
.srt-audio-cue-times {
  display: flex; gap: 0.4rem; align-items: center; flex: 1;
}
.srt-audio-cue-times input {
  width: 8.2rem; font-variant-numeric: tabular-nums;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.9rem;
}
.srt-audio-cue-chip {
  font-size: 0.78rem; opacity: 0.85;
}
.srt-audio-cue-card textarea.srt-audio-text {
  width: 100%; resize: none; overflow: hidden;
  min-height: 3.2rem; max-height: 14rem;
  line-height: 1.45; font-size: 1.02rem;
  border-radius: 8px; padding: 0.65rem 0.75rem;
}
.srt-audio-help { font-size: 0.88rem; opacity: 0.8; margin: 0.35rem 0 0; }
```

Widen editor container: change `#srt-audio-cue-editor` wrapper `max-height` to `min(70vh, 560px)` if present.

- [ ] **Step 2: Rewrite `renderSrtAudioCues` + auto-grow**

```javascript
function autoGrowSrtAudioTextarea(el) {
  if (!el) return;
  el.style.height = "auto";
  const max = 224; // ~14rem
  el.style.height = Math.min(el.scrollHeight, max) + "px";
  el.style.overflowY = el.scrollHeight > max ? "auto" : "hidden";
}
function renderSrtAudioCues(rows) {
  srtAudioCues = rows || [];
  if (!srtAudioCueEditor) return;
  if (!srtAudioCues.length) {
    srtAudioCueEditor.innerHTML = `<div class="subtitle-empty">Chưa có cue.</div>`;
    return;
  }
  srtAudioCueEditor.innerHTML = `<div class="srt-audio-cue-list">${srtAudioCues.map((cue) => {
    const blocking = cueHasBlockingIssues(cue);
    const warningOnly = cueHasWarningOnly(cue);
    const cardClass = blocking ? "has-error" : (warningOnly ? "has-warning" : "");
    const issueText = (cue.issues || []).map(issueLabel).join(", ");
    return `
      <div class="srt-audio-cue-card ${cardClass}" data-index="${cue.index}">
        <div class="srt-audio-cue-head">
          <span class="srt-audio-cue-index">#${cue.index}</span>
          <div class="srt-audio-cue-times">
            <input class="srt-audio-start" data-index="${cue.index}" value="${escapeHtml(cue.start)}" aria-label="Start #${cue.index}" />
            <span aria-hidden="true">→</span>
            <input class="srt-audio-end" data-index="${cue.index}" value="${escapeHtml(cue.end)}" aria-label="End #${cue.index}" />
          </div>
          ${issueText ? `<span class="srt-audio-cue-chip" style="color:${blocking ? "#c0392b" : "#d68910"}">${escapeHtml(issueText)}</span>` : ""}
          <span class="srt-audio-cue-chip">ước lượng ${cue.estimated_ms || 0}ms</span>
        </div>
        <textarea class="srt-audio-text" data-index="${cue.index}" rows="2">${escapeHtml(cue.text || "")}</textarea>
      </div>`;
  }).join("")}</div>`;
  srtAudioCueEditor.querySelectorAll("textarea.srt-audio-text").forEach((el) => {
    autoGrowSrtAudioTextarea(el);
    el.addEventListener("input", () => autoGrowSrtAudioTextarea(el));
  });
  updateSrtAudioSynthesizeEnabled(srtAudioCues.some((c) => cueHasBlockingIssues(c)));
}
```

Keep `collectSrtAudioCuesFromEditor` selectors (`.srt-audio-start` etc.) unchanged.

- [ ] **Step 3: Refresh cues when synth completes**

In `beginSrtAudioPolling`, inside `data.status === "completed"` block, after setting player URL:

```javascript
try {
  await refreshSrtAudioCues();
  showSrtAudioSuccess("Đã tạo audio. Timestamp đã cập nhật theo thời gian đọc thực tế.");
} catch (err) {
  showSrtAudioSuccess("Đã tạo audio thuyết minh.");
}
```

(Note: the poll callback is already `async`.)

- [ ] **Step 4: Optional help under speed control**

Under `#srt-audio-saydi-speed` field:

```html
<p class="srt-audio-help">Tốc độ do bạn chọn. Nếu câu dài hơn khoảng trống tới cue sau, hệ thống sẽ đẩy các cue sau và cập nhật timestamp (nghỉ ~280ms giữa câu).</p>
```

- [ ] **Step 5: Manual smoke (Docker rebuild if image bakes static)**

1. Upload `sample-srt.srt` (or short 2-cue fixture).
2. Confirm card UI + textarea grows when typing long text.
3. Synthesize (or unit already covered backend): after complete, cue start/end change when cascade applied.
4. Confirm speed control still works; no per-cue speed UI.

- [ ] **Step 6: Commit**

```bash
git add auto_subtitle/static/index.html
git commit -m "feat(srt-audio): modern cue cards and refresh cascade timestamps"
```

---

### Task 4: Verify suite + push for PR update

- [ ] **Step 1: Run focused tests**

```bash
python -m unittest tests.test_srt_audio_timing tests.test_srt_audio_job_service tests.test_srt_audio_web -v
```

Expected: all PASS.

- [ ] **Step 2: Push branch** (when PO asks or after local OK)

```bash
git push -u origin HEAD
```

PR #32 already open — new commits update the same PR. Do **not** merge.

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| User speed only; no auto-fill | Task 2 (single `saydi_config.speed`) |
| Cascade on measured duration | Task 1–2 |
| GAP 280ms + env | Task 2 |
| Rewrite cue timestamps | Task 2 |
| Manifest segments + shift_ms | Task 2 |
| Refresh UI after synth | Task 3 |
| Card editor + auto-grow | Task 3 |
| `too_long` warning only | Already done; unchanged |
| No voiceover tab changes | Out of scope |

## Self-review

- No TBD placeholders.
- `plan_cascade_starts` signature consistent across tasks.
- First cue keeps intro silence via `intent_start` (e.g. 3000).
- Re-synth uses updated `edited.srt` starts as new intent (by design).

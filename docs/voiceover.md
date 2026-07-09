# Thuyết minh (Voiceover) — DrakonSub

Tính năng **Thuyết minh** tạo video MP4 có lời đọc tiếng Việt (Saydi TTS) trộn với audio gốc, dựa trên file SRT lời đọc đã chuẩn bị sẵn.

Luồng tách biệt hoàn toàn với tab **Vietsub** (ASR → dịch → burn subtitle). Artifact lưu dưới `voiceover_jobs/` (hoặc `DRAKONSUB_VOICEOVER_JOBS_ROOT`).

## Yêu cầu

- `ffmpeg` trên máy chạy server
- Biến môi trường backend **`SAYDI_TTS_API_TOKEN`** (không bao giờ gửi ra frontend)
- Các biến Saydi tùy chọn: xem `.env.example`

## Biến môi trường

| Biến | Mặc định | Ghi chú |
|------|----------|---------|
| `SAYDI_TTS_API_URL` | `https://api.voice.saydi.ai/tts` | Endpoint TTS |
| `SAYDI_TTS_API_TOKEN` | *(bắt buộc)* | Chỉ backend |
| `SAYDI_TTS_MODEL` | `k2-fsa/OmniVoice` | Model Saydi |
| `SAYDI_TTS_SAMPLE` | voice sample id | Giọng đọc |
| `SAYDI_TTS_OUTPUT_FORMAT` | `wav` | Định dạng segment |
| `SAYDI_TTS_TIMEOUT_SECONDS` | `120` | Timeout mỗi request |
| `SAYDI_TTS_CONCURRENCY` | `3` | Song song TTS |
| `SAYDI_TTS_LANG` | `vi` | Ngôn ngữ |
| `DRAKONSUB_VOICEOVER_JOBS_ROOT` | `voiceover_jobs` | Thư mục job artifact |

## Probe kết nối Saydi (Phase 0)

```bash
python3 scripts/probe_saydi_tts.py
```

In trạng thái kết nối; output mẫu (nếu có) vào `voiceover_probe_output/` (đã gitignore).

## CLI prototype (Phase 1)

```bash
python3 scripts/prototype_voiceover_from_srt.py \
  --input-video sample-video.mp4 \
  --voiceover-srt tests/fixtures/final_voiceover_vi_minimal.srt \
  --output output_voiceover.mp4
```

## Web UI

1. Chạy server: `uvicorn auto_subtitle.web:app --reload --host 0.0.0.0 --port 8000`
2. Mở `http://127.0.0.1:8000`
3. Tab **Thuyết minh** → upload video + SRT → **Tạo video thuyết minh**
4. UI poll `GET /api/voiceover/jobs/{job_id}` mỗi 2 giây cho đến `completed` hoặc `failed`
5. Tải output MP4 và manifest JSON khi hoàn tất

## API

| Method | Path | Mô tả |
|--------|------|-------|
| `POST` | `/api/voiceover/jobs` | Tạo job async, trả `job_id` + `status_url` |
| `GET` | `/api/voiceover/jobs/{job_id}` | Trạng thái, progress, URL tải khi ready |
| `GET` | `/api/voiceover/jobs/{job_id}/manifest` | Manifest JSON (409 nếu chưa ready) |
| `GET` | `/api/voiceover/jobs/{job_id}/output-video` | MP4 output (409 nếu chưa ready) |

## Artifact mỗi job

```
voiceover_jobs/{job_id}/
  job.json
  input.mp4
  voiceover.srt
  prepared_voiceover.srt   # nếu bật prepare_text
  segments/
  voiceover_track.wav
  mixed_audio.wav
  output_voiceover.mp4
  manifest.json
```

Thư mục trên **không commit** (`.gitignore`).

## Smoke checklist (manual)

### Vietsub (regression)

- [ ] App load, tab Vietsub mặc định
- [ ] Upload video, URL import, topic/engine controls hiển thị
- [ ] Không vỡ layout sau khi mở tab Thuyết minh rồi quay lại

### Voiceover

- [ ] Tab Thuyết minh mở được
- [ ] Upload video + SRT, submit trả nhanh (không đơ UI)
- [ ] Progress polling hiển thị
- [ ] Completed → tải MP4 phát được, manifest tải được

### Lỗi / edge cases

- [ ] Thiếu video hoặc SRT → UI chặn / API 422
- [ ] Video sai định dạng → 400 friendly
- [ ] Volume ngoài khoảng → UI validation
- [ ] Manifest/output trước khi ready → 404/409
- [ ] Job failed → message thân thiện, không lộ token/path tuyệt đối trên UI status

## Hạn chế đã biết

1. Cần `SAYDI_TTS_API_TOKEN` ở backend.
2. Chuẩn bị text deterministic; chưa có LLM rewrite nâng cao.
3. SRT quá ngắn so với độ dài lời đọc → có thể `severe_overflow`.
4. Background thread in-process; chưa có queue bền (Redis/Celery).
5. Server restart khi job đang chạy → job không tự resume.

## Tests

```bash
python3 -m pytest -q tests/test_voiceover_web.py tests/test_voiceover_job_service.py
```

Regression web liên quan:

```bash
python3 -m pytest -q tests/test_health_web.py tests/test_url_import_web.py
```

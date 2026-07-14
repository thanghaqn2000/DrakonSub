# Design: SRT → Audio (thuyết minh từ SRT)

Date: 2026-07-14  
Branch: `feature/implement_srt_feature`  
Status: Draft for PO review

## Problem

User đã có file SRT (text + timestamp) và muốn tạo **audio thuyết minh** qua Saydi, không cần video. Flow hiện tại “Thuyết minh” gắn với video (Whisper → dịch → mix video), không phù hợp.

## Goals

- Upload SRT → xem/sửa cue trên UI → generate audio theo timeline SRT → nghe + tải về.
- Không dùng Whisper / Gemini / video mux.
- Không auto tăng speed / cascade-shift khi cue quá dài; user phải sửa timestamp.

## Non-goals

- Mix audio vào video.
- Thêm / xóa / tách / gộp cue.
- Auto-repair timing hoặc re-TTS với speed thay đổi khi overflow.
- Cải thiện chất lượng dịch SRT.

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Output | Audio only |
| UI | Tab mới, tách khỏi “Thuyết minh” |
| Editor | Chỉ sửa text + start/end của cue hiện có |
| Overflow | Block + báo lỗi; user tự sửa |
| Pre-check | Ước lượng độ dài trước khi gọi Saydi (chars / reading rate) |
| Download | WAV hoặc MP3 (user chọn) |
| Architecture | Job type mới `srt-audio`, tái sử dụng Saydi + pattern cue editor |

## User flow

1. Mở tab **SRT → Audio**.
2. Upload file `.srt`.
3. UI hiển thị danh sách cue (index, start, end, text) dễ đọc.
4. User chỉnh text và timestamp → **Lưu**.
5. User chọn giọng Saydi, tốc độ, format tải (WAV/MP3).
6. Hệ thống **ước lượng** trước: cue quá dài so với slot hoặc timestamp invalid/overlap → highlight + disable **Thuyết minh**.
7. User bấm **Thuyết minh** → gọi Saydi theo từng cue → ghép timeline (khoảng trống = silence).
8. Job xong → audio player + nút **Tải về** (theo format đã chọn; hoặc cho đổi format tải nếu đã có WAV gốc).

## Validation rules (editor + pre-synthesize)

- Text cue không được rỗng (sau trim).
- `start < end` cho mọi cue.
- Không chồng thời gian giữa cue liên tiếp (`end[i] <= start[i+1]`).
- Ước lượng: `estimated_ms ≈ f(char_count, saydi_speed, chars_per_second)`; nếu `estimated_ms > cue_duration_ms` → flag `too_long`.
- Hằng số ước lượng dùng config/env (mặc định gần với max CPS voiceover hiện tại, hiệu chỉnh theo `saydi_speed`).
- Button synthesize chỉ enable khi không còn lỗi validation / ước lượng.

Sau khi TTS (defense in depth, không thay quyết định A): nếu vẫn đo được duration thực tế > slot, ghi warning vào manifest; **không** auto-shift. (MVP có thể skip và tin pre-check; nên ghi note trong plan.)

## Backend API (proposed)

Base: `/api/srt-audio`

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/jobs` | Upload SRT, tạo job, parse cues |
| `GET` | `/jobs/{id}` | Status + metadata |
| `GET` | `/jobs/{id}/cues` | Danh sách cue |
| `PUT` | `/jobs/{id}/cues` | Lưu chỉnh sửa text + timestamps |
| `POST` | `/jobs/{id}/validate` | Optional: trả kết quả ước lượng/overlap (có thể gộp vào GET cues) |
| `POST` | `/jobs/{id}/synthesize` | Chạy Saydi + build track (async job) |
| `GET` | `/jobs/{id}/audio` | Stream/download WAV hoặc MP3 (`?format=wav\|mp3`) |

Job storage: thư mục riêng (vd. `data/srt-audio-jobs/{job_id}/`) với `job.json`, `input.srt`, `edited.srt`, `segments/`, `output.wav`, `output.mp3` (nếu cần), `manifest.json`.

## Audio pipeline

1. Parse effective SRT (edited nếu có).
2. Re-validate hard rules; reject nếu fail.
3. Pre-estimate all cues; reject synthesize nếu còn `too_long` / overlap.
4. Với mỗi cue: `synthesize_to_file` (Saydi) → WAV segment.
5. Build một track dài: `adelay` theo `start_ms` + silence padding; `duration_ms = max(end of last audio, last cue end)`.
6. Nếu user chọn MP3: ffmpeg convert từ WAV.
7. Không atruncate theo slot (vì đã block pre-check); full segment đặt tại start.

Reuse tối đa:

- `auto_subtitle.voiceover.saydi_tts`
- parse/write SRT helpers (`subtitle_edit_service` / `voiceover.srt_parser`)
- pattern job JSON + background thread giống voiceover script jobs
- ffmpeg filtergraph tương tự `build_voiceover_track` (không mux video)

## UI

- Mode tab mới bên cạnh Vietsub / Thuyết minh.
- Step 1: upload SRT.
- Step 2: cue editor (inputs start/end/text; highlight lỗi).
- Step 3: Saydi sample + speed + format + button Thuyết minh + progress.
- Step 4: `<audio controls>` + download button.
- Copy tiếng Việt, thống nhất style hiện có trong `index.html`.

## Testing

- Unit: SRT parse upload, cue PUT validation (overlap, empty text, start>=end), estimate-too-long, synthesize reject khi invalid.
- Unit/integration (mock Saydi): build timeline silence + placement; MP3 conversion path mocked/ffmpeg if available.
- Web smoke: create job → edit → synthesize status → audio ready URLs.

## Open details for implementation plan

- Exact estimate formula và default CPS (đề xuất: base ~13 chars/s @ speed 1.0, chia cho `saydi_speed`).
- Có expose endpoint `validate` riêng hay chỉ trả `warnings` trong `GET/PUT cues`.
- Khi synthesize fail giữa chừng: partial cleanup vs giữ segments để retry (đề xuất: fail sạch + message rõ).

## Success criteria

- User upload SRT, sửa được text/timestamp, không thêm/xóa cue.
- Cue quá dài (theo ước lượng) hoặc timestamp lỗi → không gọi Saydi.
- Generate xong nghe được trên UI và tải WAV hoặc MP3.
- Không đụng flow Thuyết minh video hiện tại.

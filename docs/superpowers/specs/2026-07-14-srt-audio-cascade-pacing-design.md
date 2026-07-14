# Design: SRT → Audio — cascade pacing + editor UI

Date: 2026-07-14  
Branch: `feature/implement_srt_feature`  
Status: Approved in chat; pending PO review of this file  
Related: `2026-07-14-srt-to-audio-design.md` (MVP; **superseded** on timing rigidity / overflow)

## Problem

Pipeline hiện tại đặt mỗi segment TTS đúng `start` SRT, dùng **một** `saydi_speed` global, rồi để silence đến cue sau. Hệ quả:

- Slot dài + câu vừa → nói xong sớm, nghỉ dài (cảm giác chia thời gian lệch).
- Slot ngắn + câu dài → segment có thể lấn vào vùng thời gian của cue sau (chồng audio), hoặc chỉ cảnh báo `too_long` mà không sắp xếp lại.

PO muốn audio **đọc tự nhiên theo tốc độ đã chọn**, timeline **không cứng** theo SRT gốc: thiếu chỗ thì **đẩy cue sau** (cascade), rồi **cập nhật timestamp trên UI** theo thời gian thật. Không tự giảm/tăng speed để lấp slot.

Ngoài ra editor cue vẫn dạng input mặc định, khó đọc — cần UI hiện đại hơn, textarea tự phóng theo nội dung.

## Goals

- Giữ tốc độ Saydi **do user chọn** (default `1.0`, min/max như config hiện có); **không** auto điều chỉnh speed per cue để fill slot.
- Khi duration TTS thật khiến cue kết thúc sát / đụng cue kế → **cascade-shift** `planned_start` của các cue sau, chừa **GAP nghỉ ngắn** giữa hai câu.
- Sau synthesize: persist và hiển thị `actual_start` / `actual_end` trên editor (timeline cascade).
- Editor cue: layout card hiện đại, textarea auto-grow.
- Output audio có thể **dài hơn** tổng timeline SRT gốc — OK (dùng ghép video khác).

## Non-goals

- Auto slow-down / speed-up per cue để khớp `end` SRT (PO chọn phương án B).
- Time-stretch audio sau TTS hoặc chèn pause trong câu.
- Đổi pipeline Voiceover gắn video (PR #31 / tab Thuyết minh).
- Thêm / xóa / merge cue.
- Visual companion / redesign toàn app — chỉ editor SRT → Audio (+ CSS scoped).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Base speed | UI global, default `1.0`, user chỉnh tay |
| Fill slot dư | **Không** — giữ tốc độ user; silence còn lại chấp nhận |
| Thiếu thời gian | Cascade theo **duration đo được** sau TTS (hướng 2) |
| GAP giữa cue | **280ms** (`prev_end + 280` làm floor cho `planned_start` tiếp) |
| Timestamp sau synth | Cập nhật cues trên UI + disk theo timeline cascade |
| `too_long` pre-check | Chỉ **cảnh báo** (không block); cascade xử lý thiếu chỗ |
| Speeding khi overflow | Không bắt buộc tăng speed; để cascade đẩy cue sau |
| Editor UI | Card + textarea auto-grow; time compact |

## Placement algorithm

Với mỗi cue theo thứ tự index:

1. TTS với `saydi_speed` global → file segment → `duration_ms` (probe thật).
2. `intent_start_ms` = `start` đang lưu trên cue (SRT / chỉnh tay trước synth).
3. `planned_start_ms = max(intent_start_ms, prev_end_ms + GAP_MS)` với `GAP_MS = 280`, `prev_end_ms` của cue đầu = `0` (hoặc không áp floor nếu là cue đầu và `intent_start` đã đủ).
4. `planned_end_ms = planned_start_ms + duration_ms`.
5. Ghi vào manifest + cập nhật cue: `start`/`end` (hoặc field `actual_*` rồi sync vào `start`/`end` hiển thị) = timeline cascade.
6. Cue kế dùng `prev_end_ms = planned_end_ms`.

Ghép track: `adelay` theo `planned_start_ms` (không truncate theo `end` SRT cũ).

**Cue đầu:** nếu `intent_start > 0`, vẫn tôn trọng intro silence (intro nhạc/im đầu video đích).

## Data model

- Trước synth lần đầu: cues = ý định user / SRT gốc.
- Sau synth thành công:
  - Cues trên job được **ghi đè** `start`/`end` theo cascade (PO: cập nhật UI theo cascade).
  - Manifest mỗi segment: `index`, `text`, `intent_start_ms`, `planned_start_ms`, `duration_ms`, `planned_end_ms`, `shift_ms` (= `planned_start - intent_start`, có thể 0).
- Optional (khuyến nghị, không bắt buộc UI): giữ `original_start_ms` / `original_end_ms` trong manifest để debug; không cần hiện trên editor trừ khi sau này PO muốn badge “đã dịch +Xms”.

Re-synth: dùng `start` hiện tại trên cues (đã cascade) làm `intent_start` mới — hành vi ổn định, không nhảy về SRT gốc trừ khi user upload lại / restore (không có restore trong scope).

## Validation (cập nhật)

| Issue | Trước synth | Ghi chú |
|-------|-------------|---------|
| `empty_text`, `start_after_end`, `invalid_timestamp` | **Block** | Giữ |
| `overlap_next` (theo số trên editor) | **Block** | User sửa trước khi synth; sau cascade timeline không overlap |
| `too_long` | **Warning only** | Không block; cascade xử lý |

## UI changes

### Editor

- Mỗi cue: card (index badge, start/end compact, textarea text).
- Textarea: `auto-resize` theo nội dung (min ~2–3 dòng, max chiều cao hợp lý + scroll nội bộ nếu cực dài).
- Warning/error chip vẫn hiện trên card (màu phân biệt warning vs blocking).
- Sau job `completed`: `GET cues` trả timestamp mới → `renderSrtAudioCues` refresh; có thể hiện dòng phụ ngắn: “Đã căn theo audio thực tế” (optional copy).

### Synthesize panel

- Giữ control tốc độ như hiện tại (không đổi thành “auto only”).
- Copy/help ngắn (optional): thiếu thời gian sẽ đẩy cue sau; không tự đổi tốc độ.

## Backend touchpoints

- `auto_subtitle/srt_audio/job_service.py` — vòng TTS + đặt `planned_start` cascade; ghi lại cues.
- `auto_subtitle/srt_audio/audio_track.py` — dùng `planned_start` (đã có `start_ms` từ caller).
- `auto_subtitle/srt_audio/cue_service.py` / API GET-PUT — trả/nhận timestamp đã cập nhật; không đổi contract lớn.
- Config: `SRT_AUDIO_CUE_GAP_MS` (default 280) trong config + `.env.example`.
- `static/index.html` — CSS + `renderSrtAudioCues` + auto-grow helpers.

## Testing

- Unit: placement helper — `planned_start = max(intent, prev_end+gap)`; chuỗi overflow đẩy cue 2, 3…
- Unit: sau synth mock, cues file có start/end cascade, không overlap, gap ≥ 280ms giữa end[i] và start[i+1].
- Unit: cùng speed global; không ghi per-cue speed khác user.
- UI smoke (manual): editor card + textarea grow; sau synth timestamp đổi.

## Success criteria

- Speed chỉ theo UI user; không auto fill bằng slow-down.
- Cue dài hơn khoảng trống tới cue sau → cue sau (và chuỗi) bị đẩy; audio không chồng lời.
- Giữa hai câu luôn có ~280ms nghỉ (trừ khi intent_start đã cách xa hơn).
- Editor hiển thị timestamp cascade sau synth; textarea dễ đọc, auto phóng.
- Không phá tab Vietsub / Thuyết minh video.

## Self-review notes

- Không còn mâu thuẫn với “block too_long / no cascade” của spec MVP — file này **supersede** các quyết định timing đó.
- Không auto-slow (PO chọn B) dù triệu chứng ban đầu là “đọc nhanh nghỉ lâu”; PO chấp nhận silence khi slot dư nếu giữ speed tay.
- GAP 280ms cố định; có thể chỉnh qua env nếu cần sau.

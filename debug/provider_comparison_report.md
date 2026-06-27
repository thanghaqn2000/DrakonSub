# Báo cáo chẩn đoán: Gemini 2.5 Flash vs OpenAI gpt-4o-mini

**Ngày:** 2026-06-27  
**Phạm vi:** Chỉ chẩn đoán — không refactor, không đổi hành vi pipeline (ngoài script debug + dump prompt).  
**Mẫu test:** Buffett/Bitcoin (~29 cue) — `source.srt` từ job `6bfeabd2-…`

---

## Executive summary

Gemini 2.5 Flash thường cho subtitle tiếng Việt tốt hơn **không phải vì pipeline post-processing khác**, mà chủ yếu vì:

1. **Raw translation prompt & context khác hẳn OpenAI** — Gemini có `previous_context` / `next_context` ±5 cue và system prompt kiểu *localization editor*; OpenAI chỉ gom phrase group trong batch, **không** có prev/next context ngoài batch.
2. **Model khác nhau ở bước quan trọng nhất** — `TRANSLATION_ENGINE=gemini` → raw + editor dùng **Gemini 2.5 Flash**; `TRANSLATION_ENGINE=openai` → raw + editor dùng **gpt-4o-mini** (từ `OPENAI_MODEL` trong `.env`).
3. **Editor pass dùng cùng prompt** nhưng trên gpt-4o-mini **hầu như không sửa** các cue vấn đề B–E; chất lượng đã kém từ raw và giữ nguyên qua editor.
4. **Compression + Flow luôn gọi OpenAI** (`get_openai_model()`), không theo `TRANSLATION_ENGINE` — cả hai provider đều qua cùng pass này; với OpenAI 4o-mini, flow pass **cứu được** nhiều cue B/C/D/E ở cuối pipeline.
5. **Không có Final QA pass** trong codebase — stage `after_final_qa` chỉ là bản copy sau flow.
6. **Bug empty cue (OpenAI):** cue 5 (`I do with it?`) trống từ **raw translation** — model gộp nghĩa vào cue 4, trả chuỗi rỗng cho cue 5; parser chấp nhận và `openai_translate.py` ghi `""` thay vì fallback.

**Artifact Gemini:** không chạy xong do **Gemini API quota 429** (free tier 20 req/ngày). Chỉ có `debug/provider_comparison/gemini/gemini_source.srt`. Artifact OpenAI đầy đủ tại `debug/provider_comparison/openai/`.

---

## 1. Pipeline comparison (side-by-side)

Cả hai provider (EN → VI) đi qua `generate_vietsub()` → nhánh `source_language != "vi"`.

| # | Stage | Gemini (`TRANSLATION_ENGINE=gemini`) | OpenAI (`TRANSLATION_ENGINE=openai`) |
|---|--------|--------------------------------------|--------------------------------------|
| 0 | ASR / source | `transcribe_to_srt` → `source.srt` | Giống |
| 1 | EN domain correction | `en_domain_corrector.correct_en_domain_srt_file` | Giống |
| 2 | **Raw translation** | `utils.translate_srt_entries` → **`gemini_translate.translate_srt_entries_gemini`** | → **`openai_translate.translate_srt_entries_openai`** |
| 3 | Inline OpenAI polish | **Tắt** (`translation_polish_enabled()` → `False`) | **Tắt** |
| 4 | **VI Editor** | `vi_editor.edit_vi_srt_file` — provider **`gemini`** (`VI_EDITOR_PROVIDER=auto`) | `vi_editor` — provider **`openai`** |
| 5 | **VI Compression** | `vi_compression.compress_vi_srt_file` — **luôn OpenAI** (`get_openai_model()`) | Giống |
| 6 | **Multi-cue Flow** | `vi_flow.flow_vi_srt_file` — **luôn OpenAI** | Giống |
| 7 | Final QA / repair | **Stage missing** — không có module | **Stage missing** |
| 8 | Readability | `subtitle_readability_optimizer.optimize_readability_file` (rule-based; OpenAI opt-in qua env, mặc định **false**) | Giống |
| 9 | Timing optimizer | `subtitle_timing_optimizer.optimize_srt_timing_file` | Giống |
| 10 | Timing normalize | `subtitle_timing_optimizer.normalize_final_srt_timing` | Giống |
| 11 | Renderer | `subtitle_renderer.burn_subtitles` | Giống |

### Khác biệt đáng chú ý

| Khác biệt | Chi tiết |
|-----------|----------|
| Raw translation module | `gemini_translate.py` vs `openai_translate.py` |
| Editor LLM | Gemini 2.5 Flash vs gpt-4o-mini |
| Compression / Flow LLM | **Cả hai engine đều dùng `OPENAI_MODEL`** (hiện tại `gpt-4o-mini`) — **không** dùng Gemini |
| OpenAI-only polish | Code còn `_polish_translations` trong `openai_translate.py` nhưng **bị vô hiệu** trong config |
| VI path (`source_language=vi`) | Bỏ qua translate/editor/compression/flow; chỉ `vi_loanword_fix` |

---

## 2. Prompt comparison

Prompt dump: `debug/provider_comparison/prompts/`

### Raw translation

| | OpenAI | Gemini |
|---|--------|--------|
| **System** | `translation_topics.build_system_prompt()` — “professional Vietnamese subtitle **translator**” + topic economics + glossary | `_GEMINI_SYSTEM_PROMPT` — “**localization editor**”, rewrite meaning, không dịch từng từ + `_DOMAIN_GLOSSARY` + topic |
| **User** | `_build_grouped_user_prompt()` — phrase groups trong batch, **không** prev/next context | `_build_grouped_user_prompt()` — **`previous_context` (5 cue)**, `current_batch`, **`next_context` (5 cue)** |
| **Giống nhau?** | **Không** | |
| **EN + VI draft?** | Chỉ EN | Chỉ EN |
| **Few-shot** | Không | Không |
| **JSON** | `{"translations": [...]}` + `response_format json_object` | `{"translations": [...]}` + `responseMimeType: application/json` |

### Vietnamese Editor

| | OpenAI | Gemini |
|---|--------|--------|
| **System** | `VI_EDITOR_SYSTEM_PROMPT` trong `vi_editor.py` | **Cùng file, cùng prompt** |
| **User** | EN source + VI raw + **±5 cue** prev/next context (`_build_editor_user_prompt`) | **Giống hệt** |
| **Few-shot** | Có (BAD/GOOD, multi-cue Buffett) | Có |
| **JSON** | `{"items": [{"index", "text_vi"}]}` | Giống |

### Compression

| | OpenAI | Gemini |
|---|--------|--------|
| Provider | **Chỉ OpenAI** | **Stage không gọi Gemini** |
| Prompt | `_COMPRESSION_SYSTEM_PROMPT` + user batch có duration/CPS | N/A |

### Multi-cue Flow

| | OpenAI | Gemini |
|---|--------|--------|
| Provider | **Chỉ OpenAI** | **Stage không gọi Gemini** |
| Prompt | `_FLOW_SYSTEM_PROMPT` + 4 few-shot + EN/VI/duration | N/A |

### Final QA

**Stage missing for provider: OpenAI**  
**Stage missing for provider: Gemini**

---

## 3. Model settings comparison

| Setting | OpenAI (raw + editor khi `TRANSLATION_ENGINE=openai`) | Gemini (raw + editor khi `TRANSLATION_ENGINE=gemini`) | Compression / Flow (cả hai engine) |
|---------|------------------------------------------------------|------------------------------------------------------|-----------------------------------|
| Model | `OPENAI_MODEL` → **gpt-4o-mini** (`.env`) | `GEMINI_MODEL` → **gemini-2.5-flash** | `get_openai_model()` → **gpt-4o-mini** |
| Temperature (raw) | 0.4 | 0.4 | 0.2–0.25 (compress), 0.25 (flow) |
| Temperature (editor) | 0.3 (`DEFAULT_VI_EDITOR_TEMPERATURE`) | 0.3 | — |
| max_tokens | Không set (API default) | Không set trong payload | Không set |
| top_p / top_k | Không set | Không set | Không set |
| Response format | `json_object` (Chat Completions) | `application/json` (Gemini `generateContent`) | `json_object` |
| Timeout | OpenAI SDK default | 120s (`urllib` `urlopen`) | OpenAI SDK default |
| Retry (raw) | Batch → group → per-cue legacy | Batch retry max **2**, rồi group retry | Batch + strict retry (compress/flow) |
| Batch size | `TRANSLATION_BATCH_SIZE` = **30** | **30** | Editor **30**; compress **20**; flow theo nhóm 2–4 |
| Phrase group max | `PHRASE_GROUP_MAX_CUES` = **6** | **6** | — |
| Editor context window | **5** cues | **5** cues | — |
| Provider cleanup | `_parse_json_strings`; empty → `""` (raw) | `_parse_json_strings`; missing → **fallback EN gốc** | Rule-based + validation |

---

## 4. Artifact comparison

Script: `scripts/run_provider_comparison.py`  
Output root: `debug/provider_comparison/`

### OpenAI — đầy đủ

| File | Path |
|------|------|
| source | `openai/openai_source.srt` |
| raw | `openai/openai_raw_translation.srt` |
| after editor | `openai/openai_after_editor.srt` |
| after compression | `openai/openai_after_compression.srt` |
| after flow | `openai/openai_after_flow.srt` |
| after final qa | `openai/openai_after_final_qa.srt` *(bản copy sau flow — không có QA LLM)* |
| final | `openai/openai_final.srt` |
| JSON theo dõi cue | `openai/openai_comparison.json` |

### Gemini — không đầy đủ (quota 429)

| File | Trạng thái |
|------|-----------|
| gemini_source.srt | ✅ Có |
| gemini_raw_translation.srt | ❌ **Stage missing** — API quota exceeded trước khi dịch xong |
| gemini_after_editor.srt | ❌ **Stage missing** |
| gemini_after_compression.srt | ❌ **Stage missing** |
| gemini_after_flow.srt | ❌ **Stage missing** |
| gemini_after_final_qa.srt | ❌ **Stage missing** |
| gemini_final.srt | ❌ **Stage missing** |

Chạy lại khi quota reset: `python3 scripts/run_provider_comparison.py` (bỏ `SKIP_GEMINI_COMPARISON=1`).

---

## 5. OpenAI quality drops — theo từng stage (Buffett sample)

Nguồn: `debug/provider_comparison/openai/openai_comparison.json`

### Case A — farmland (không có trong mẫu Buffett)

EN *"If you said for a 1% interest in all the farmland…"* **không xuất hiện** trong `source.srt` mẫu này. Cần sample/video khác để trace. Hướng mong đợi (*"Nếu anh chào bán 1% toàn bộ đất nông nghiệp…"*) phụ thuộc raw translation + editor — chưa đo được trên artifact hiện tại.

### Case B — cue 7–8 (*same people / do anything*)

| Stage | Cue 7 | Cue 8 |
|-------|-------|-------|
| Raw | `Ý tôi là, có thể tôi sẽ có những người giống nhau, nhưng` | `nó sẽ chẳng mang lại điều gì cả.` |
| Editor | **Không đổi** | **Không đổi** |
| Compression | `Có thể vẫn có người giống nhau,` | `Nó sẽ không mang lại gì.` |
| Flow | ✅ `Có thể vẫn có người mua lại, nhưng` | ✅ `bản thân nó chẳng tạo ra gì.` |

**Kết luận:** Đã kém / literal từ **raw**; **editor không sửa**; compression cải thiện một phần; **flow sửa đúng hướng**.

### Case C — cue 14–15 (*mystery / about it*)

| Stage | Cue 14 | Cue 15 |
|-------|--------|--------|
| Raw | `Nếu tôi có tất cả, anh ấy có thể tạo ra một bí ẩn` | `về điều đó.` |
| Editor | **Không đổi** | **Không đổi** |
| Compression | `Nếu tôi giữ hết, nó vẫn chỉ là bí ẩn.` | `về điều đó.` *(vẫn fragment)* |
| Flow | ✅ `Nếu tôi giữ hết, nó vẫn chỉ là` | ✅ `một điều bí ẩn.` |

**Kết luận:** Fragment `về điều đó.` từ **raw**; editor bỏ qua; compression gom sai vào cue 14; **flow sửa**.

### Case D — cue 24–25 (*anything for / it*)

| Stage | Cue 24 | Cue 25 |
|-------|--------|--------|
| Raw | `Nhưng tôi sẽ không cho bạn bất cứ điều gì cả` | `về chuyện đó.` |
| Editor | **Không đổi** | **Không đổi** |
| Compression | **Không đổi** | **Không đổi** |
| Flow | ✅ `Nhưng bảo tôi bỏ tiền mua` | ✅ `thì không có đâu.` |

**Kết luận:** Stiff từ **raw**; editor/compression không cứu; **chỉ flow** đạt good VI.

### Case E — cue 27–29 (*productive assets / greater fool*)

| Stage | Tóm tắt |
|-------|---------|
| Raw / Editor / Compression | Dài, literal: *"điều đó giải thích…", "người tiếp theo", "trả cho bạn nhiều hơn…"* — **không đổi** qua editor & compression |
| Flow | ✅ `Đó là khác biệt… tạo ra giá trị` / `chờ người sau` / `mua lại với giá cao hơn.` |

### Tổng hợp “điểm rơi chất lượng” (OpenAI 4o-mini)

```
Raw translation  ████████████  (literal, merge cue, fragment)
VI Editor        ██████████    (gần như pass-through trên case B–E)
Compression      ████          (một số cue; có thể tạo fragment mới — cue 15)
Flow             ██            (cứu cross-cue — nhưng dùng cùng 4o-mini + rules)
Final QA         —             (không tồn tại)
Readability/Timing  ~            (chủ yếu CPS/timing, không sửa nghĩa lớn)
```

---

## 6. Parser / empty cue bug

**Hiện tượng:** Cue **5** trống ở mọi stage OpenAI (`empty_cues_by_stage.raw_translation: [5]`).

**EN:**
- Cue 4: `for $25, I wouldn't take it because what would`
- Cue 5: `I do with it?`

**VI raw:**
- Cue 4: `tôi cũng sẽ không nhận đâu, vì tôi sẽ làm gì với nó?` *(gộp cả cue 5)*
- Cue 5: *(trống)*

**Nguyên nhân (theo thứ tự):**

1. **Model trả `""` cho cue 5** trong JSON `translations` (đúng count 29 nhưng cue 5 rỗng).
2. **`openai_translate.py` ~L536:** `translated_non_empty.get(local_idx, "")` — thiếu key → **chuỗi rỗng**, không fallback EN (khác Gemini ~L473 dùng fallback text gốc).
3. **`vi_editor.py` ~L368–376:** Nếu raw rỗng → giữ `""` (đúng rule “empty unless source empty” — nhưng EN cue 5 **không** empty).
4. **`utils.parse_srt`:** Chấp nhận cue text rỗng (đã fix trước đó để giữ cue count).
5. **Validation:** Không có bước reject cue VI trống khi EN có text.

**File/function chịu trách nhiệm chính:** `openai_translate.translate_srt_entries_openai` (reconstruct + default `""`); gốc vấn đề là **model merge + empty string** trong batch translation.

---

## 7. OpenAI-specific / provider branches

| Vị trí | Mô tả | Ảnh hưởng hiện tại |
|--------|--------|-------------------|
| `openai_translate._polish_translations` | Polish chỉ VI, **không** có EN source | **Tắt** (`translation_polish_enabled() → False`) |
| `openai_translate` vs `gemini_translate` | Prompt & user context **khác** | **Cao** — nguồn chênh lệch chính |
| `vi_editor` `resolve_vi_editor_provider` | `auto` → theo `TRANSLATION_ENGINE` | Editor model khác nhau |
| `vi_compression` / `vi_flow` | **Chỉ** `get_openai_model()` | Gemini path vẫn post-process bằng **4o-mini** |
| `utils.translate_srt_entries` | Router `openai` \| `gemini` | Chỉ tầng raw |
| `subtitle_readability_optimizer` | `_openai_rewrite_batch` nếu `VI_READABILITY_USE_OPENAI=true` | Mặc định **false** |
| `vi_loanword_fix` | OpenAI optional | Chỉ nhánh `source_language=vi` |
| `openai_chat.supports_custom_temperature` | gpt-5/o-series bỏ temperature | **4o-mini** vẫn dùng temperature |

**Không thấy:** OpenAI bypass `vi_editor`, bypass compression/flow, hay path polish cũ đang chạy song song.

---

## 8. OpenAI có đi cùng post-processing pipeline với Gemini không?

**Có — gần như hoàn toàn.** Sau raw translation, `pipeline.py` gọi cùng chuỗi: editor → compression → flow → readability → timing → normalize → render.

**Ngoại lệ:** LLM backend từng bước:
- Editor: theo engine
- Compression + Flow: **luôn OpenAI** (`OPENAI_MODEL`)

Vì vậy nếu anh so sánh “Gemini job” vs “OpenAI job” trên UI:
- Gemini có thể **đã tốt từ raw/editor (Flash)**
- OpenAI path **vẫn bị 4o-mini** ở raw/editor, rồi được flow/compression (cũng 4o-mini) vá phần nào

---

## 9. Root cause hypothesis (xếp hạng)

| # | Giả thuyết | Độ tin cậy | Bằng chứng |
|---|------------|------------|------------|
| 1 | **Raw translation prompt + ±5 context của Gemini** tốt hơn OpenAI grouped prompt không prev/next | **Cao** | Code diff `gemini_translate._build_grouped_user_prompt` vs `openai_translate._build_grouped_user_prompt`; system prompt khác framing |
| 2 | **Model capability: gemini-2.5-flash > gpt-4o-mini** cho tiếng Việt subtitle | **Cao** | User quan sát; editor 4o-mini pass-through trên B–E |
| 3 | **Editor cùng prompt nhưng model yếu hơn** trên OpenAI path | **Cao** | `openai_comparison.json`: editor không đổi cue B–E |
| 4 | **OpenAI raw merge/split sai** (empty cue 5) | **Trung bình** | Artifact raw; fallback `""` vs Gemini fallback EN |
| 5 | Pipeline bypass / polish cũ | **Thấp** | Polish tắt; cùng pipeline.py |
| 6 | Compression/flow chỉ OpenAI làm Gemini kém hơn | **Thấp** | User nói Gemini **tốt hơn**; compression/flow dùng chung cho cả hai |

---

## 10. Minimal fixes đề xuất (chưa implement)

1. **Thống nhất raw user prompt** — port `previous_context` / `next_context` ±5 từ Gemini sang OpenAI (hoặc shared builder).
2. **Thống nhất system prompt** — một `build_raw_translation_system_prompt()` dùng chung, hoặc align OpenAI sang framing “localization editor”.
3. **Empty cue guard** — sau raw OpenAI: nếu EN non-empty mà VI `""` → retry cue đó hoặc fallback EN / copy từ cue kề.
4. **OpenAI reconstruct fallback** — đổi `get(local_idx, "")` → `get(local_idx, original_en)` giống Gemini (tạm thời) + metric log.
5. **Model-agnostic post-process** — `vi_compression` / `vi_flow` nên `resolve_provider(translation_engine)` thay vì hardcode OpenAI (hiện Gemini job vẫn bị 4o-mini ở hai pass này).
6. **Nâng `OPENAI_MODEL`** cho raw+editor (vd. gpt-4o / gpt-5.x) nếu giữ OpenAI path — tách env `OPENAI_TRANSLATION_MODEL` vs `OPENAI_POSTPROCESS_MODEL`.
7. **Chạy lại `run_provider_comparison.py`** khi Gemini quota available để có diff artifact đầy đủ.

---

## Phụ lục: Lệnh tái tạo

```bash
# Prompt dump only
python3 scripts/run_provider_comparison.py  # sẽ fail Gemini nếu quota hết

# OpenAI only
SKIP_GEMINI_COMPARISON=1 OPENAI_MODEL=gpt-4o-mini python3 scripts/run_provider_comparison.py
```

Prompt files: `debug/provider_comparison/prompts/*.txt`

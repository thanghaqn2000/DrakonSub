# OpenAI API — Dịch subtitle EN → VI (DrakonSub)

Tài liệu mô tả toàn bộ luồng gọi OpenAI để **dịch subtitle** trong app DrakonSub.

---

## Luồng gọi

```
pipeline.translate_srt_file()
  → utils.translate_srt_entries(engine="openai")
    → openai_translate.translate_srt_entries_openai()   # gom batch 15 segment
      → _translate_batch()
        → _call_openai_translate()
          → openai_chat.create_chat_completion()
            → client.chat.completions.create()   # POST v1/chat/completions
```

**File liên quan:**

| File | Vai trò |
|------|---------|
| `auto_subtitle/openai_translate.py` | Logic dịch, batch, parse JSON |
| `auto_subtitle/openai_chat.py` | Wrapper `chat.completions.create`, xử lý temperature |
| `auto_subtitle/translation_topics.py` | System prompt theo topic |
| `auto_subtitle/config.py` | Đọc `OPENAI_MODEL` từ `.env` |
| `auto_subtitle/utils.py` | Router `translate_srt_entries()` |
| `auto_subtitle/pipeline.py` | Gọi dịch trong pipeline EN |

---

## 1. Model

Lấy từ `.env` → biến `OPENAI_MODEL`, mặc định `gpt-5.5-2026-04-23`.

```python
# auto_subtitle/config.py
def get_openai_model() -> str:
    load_env()
    raw = (os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL).strip()
    return OPENAI_MODEL_ALIASES.get(raw, raw)
```

**Cấu hình `.env`:**

```
OPENAI_MODEL=gpt-5.5-2026-04-23
```

**Model đã verify hoạt động với Chat Completions:**

| Model | Ghi chú |
|-------|---------|
| `gpt-5.5-2026-04-23` | Cao nhất — khuyên dùng |
| `gpt-5.5` | Tương đương bản trên |
| `o3` | Reasoning, chậm & đắt |
| `gpt-4.1` | Tốt, rẻ hơn |
| `gpt-4o` | Ổn định |
| `o4-mini` | Rẻ, nhanh |

**Không dùng được:** `gpt-5.5-pro` (chỉ Responses API, không phải Chat Completions).

---

## 2. System prompt

Ghép từ `build_system_prompt(topic)` trong `translation_topics.py`.

Topic chọn từ web dropdown hoặc `.env` → `TRANSLATION_TOPIC` (`economics` | `everyday` | `humor`).

```python
# auto_subtitle/translation_topics.py
def build_system_prompt(topic: Optional[str] = None) -> str:
    topic_id = normalize_topic(topic)
    topic_def = TOPICS[topic_id]
    return (
        "You translate English video subtitles into Vietnamese.\n\n"
        f"Topic / tone: {topic_def.label}\n"
        f"{topic_def.guidance}\n\n"
        f"{_BASE_RULES}"
    )
```

**`_BASE_RULES` (chung cho mọi topic):**

```
Rules (strict):
- Translate each segment faithfully from the English source. Do not omit ideas, add ideas, or change meaning.
- One English segment → exactly one Vietnamese segment, same order.
- Do not merge or split segments.
- Keep names, numbers, and proper nouns accurate.
- Return JSON only, no markdown.
```

### Ví dụ đầy đủ — topic `economics` (mặc định)

```
You translate English video subtitles into Vietnamese.

Topic / tone: Kinh tế
Audience: general viewers, including people outside economics/finance. Content is often about economics.
- Use natural, easy-to-understand Vietnamese. Slightly colloquial/friendly is fine.
- Explain economics terms in plain language when needed; avoid stiff literal or academic wording.

Rules (strict):
- Translate each segment faithfully from the English source. Do not omit ideas, add ideas, or change meaning.
- One English segment → exactly one Vietnamese segment, same order.
- Do not merge or split segments.
- Keep names, numbers, and proper nouns accurate.
- Return JSON only, no markdown.
```

### Topic `everyday`

```
You translate English video subtitles into Vietnamese.

Topic / tone: Tự nhiên đời thường
Audience: general viewers watching everyday-life content (vlogs, stories, daily tips, education).
- Use natural spoken Vietnamese as people talk in daily life.
- Avoid economics jargon unless the source explicitly uses it; prefer plain, relatable wording.
- Keep a warm, clear tone — easy to follow on mobile.

Rules (strict):
...
```

### Topic `humor`

```
You translate English video subtitles into Vietnamese.

Topic / tone: Hài hước gần gũi
Audience: general viewers watching light, funny, or casual content.
- Use friendly, playful Vietnamese that feels close and conversational.
- Preserve humor and timing when possible; mild colloquial flair is welcome.
- Do not invent jokes or exaggerate — stay faithful to the original meaning and energy.

Rules (strict):
...
```

---

## 3. User prompt

```python
# auto_subtitle/openai_translate.py
def _build_user_prompt(segments: List[str], target_lang: str) -> str:
    lines = [f"[{i + 1}] {text}" for i, text in enumerate(segments)]
    n = len(segments)
    return (
        f"Translate these {n} English subtitle segments to {target_lang}.\n\n"
        + "\n".join(lines)
        + f'\n\nRespond with JSON: {{"translations": ["...", ...]}} '
        f"containing exactly {n} strings in the same order."
    )
```

### Ví dụ (3 segment)

```
Translate these 3 English subtitle segments to vi.

[1] The stock market rallied today.
[2] Investors are watching interest rates.
[3] The VN-Index gained 2 percent.

Respond with JSON: {"translations": ["...", ...]} containing exactly 3 strings in the same order.
```

---

## 4. Temperature

Code đặt **`temperature=0.3`**:

```python
# auto_subtitle/openai_translate.py — _call_openai_translate()
response = create_chat_completion(
    client,
    model,
    messages=[...],
    temperature=0.3,
    response_format={"type": "json_object"},
)
```

**Lưu ý:** Với model reasoning (`gpt-5.x`, `o1`, `o3`, `o4`), `temperature` **bị bỏ qua** — chỉ dùng default của model:

```python
# auto_subtitle/openai_chat.py
def supports_custom_temperature(model: str) -> bool:
    name = model.lower().split("/")[-1]
    return not (
        name.startswith("gpt-5")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    )

def create_chat_completion(client, model, messages, *, temperature=None, **kwargs):
    params = {"model": model, "messages": messages, **kwargs}
    if temperature is not None and supports_custom_temperature(model):
        params["temperature"] = temperature
    return client.chat.completions.create(**params)
```

→ Với **`gpt-5.5-2026-04-23`**: request **không có** field `temperature`.

---

## 5. max_tokens

**Không set** — app không truyền `max_tokens` hay `max_completion_tokens`. OpenAI dùng giới hạn mặc định của model.

---

## 6. response_format

```python
response_format={"type": "json_object"}
```

Bắt buộc model trả JSON. Response được parse:

```python
# auto_subtitle/openai_translate.py
def _parse_translations(content: str, expected_count: int) -> List[str]:
    data = json.loads(content)
    translations = data.get("translations", ...)
    if len(translations) != expected_count:
        raise ValueError(...)
    return [str(t).strip() for t in translations]
```

**Format JSON mong đợi:**

```json
{
  "translations": [
    "Bản dịch segment 1",
    "Bản dịch segment 2",
    "..."
  ]
}
```

Số phần tử trong `translations` phải **bằng đúng** số segment gửi lên.

---

## 7. Cách gửi dữ liệu

1. Parse file SRT → list `entries`, mỗi entry: `{ start_str, end_str, text }`
2. Lấy `text` từng segment (bỏ qua segment rỗng)
3. Gom **tối đa 15 segment** thành 1 batch
4. Gửi **chỉ text** (không gửi timestamp) trong user prompt
5. Nhận JSON `translations[]`, map lại vào entry theo thứ tự
6. Giữ nguyên `start_str`, `end_str` — chỉ thay `text`

```python
# auto_subtitle/openai_translate.py — translate_srt_entries_openai()
for i, entry in enumerate(entries):
    text = entry["text"].strip()
    if not text:
        translated[i] = {**entry, "text": text}
        continue

    pending_indices.append(i)
    pending_texts.append(text)

    if len(pending_texts) >= batch_size:
        flush_batch()   # gọi API
```

---

## 8. Mỗi lần gửi bao nhiêu segment?

**Mặc định: 15 segment / 1 API call** (`batch_size=15`):

```python
def translate_srt_entries_openai(
    entries: List[dict],
    target_lang: str = "vi",
    model: Optional[str] = None,
    batch_size: int = 15,   # ← đây
    topic: Optional[str] = None,
) -> List[dict]:
```

**Retry khi parse lỗi:** chia đôi batch và gọi lại (đệ quy):

```python
def _translate_batch(...):
    try:
        return _call_openai_translate(...)
    except ValueError:
        if len(segments) == 1:
            raise
        mid = len(segments) // 2
        return (
            _translate_batch(..., segments[:mid], ...)
            + _translate_batch(..., segments[mid:], ...)
        )
```

Ví dụ: batch 15 lỗi → 8 + 7 → nếu vẫn lỗi tiếp tục chia đôi.

---

## 9. Ví dụ payload gửi lên API

**Endpoint:** `POST https://api.openai.com/v1/chat/completions`

**Headers:**

```
Authorization: Bearer <OPENAI_API_KEY>
Content-Type: application/json
```

**Body** (model `gpt-5.5-2026-04-23`, batch 3 segment, topic `economics`):

```json
{
  "model": "gpt-5.5-2026-04-23",
  "messages": [
    {
      "role": "system",
      "content": "You translate English video subtitles into Vietnamese.\n\nTopic / tone: Kinh tế\nAudience: general viewers, including people outside economics/finance. Content is often about economics.\n- Use natural, easy-to-understand Vietnamese. Slightly colloquial/friendly is fine.\n- Explain economics terms in plain language when needed; avoid stiff literal or academic wording.\n\nRules (strict):\n- Translate each segment faithfully from the English source. Do not omit ideas, add ideas, or change meaning.\n- One English segment → exactly one Vietnamese segment, same order.\n- Do not merge or split segments.\n- Keep names, numbers, and proper nouns accurate.\n- Return JSON only, no markdown."
    },
    {
      "role": "user",
      "content": "Translate these 3 English subtitle segments to vi.\n\n[1] The stock market rallied today.\n[2] Investors are watching interest rates.\n[3] The VN-Index gained 2 percent.\n\nRespond with JSON: {\"translations\": [\"...\", ...]} containing exactly 3 strings in the same order."
    }
  ],
  "response_format": {
    "type": "json_object"
  }
}
```

**Response mong đợi:**

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"translations\": [\"Thị trường chứng khoán hôm nay tăng điểm.\", \"Nhà đầu tư đang theo dõi lãi suất.\", \"VN-Index tăng 2 phần trăm.\"]}"
      },
      "finish_reason": "stop"
    }
  ]
}
```

---

## Tóm tắt tham số

| Tham số | Giá trị |
|---------|---------|
| **Endpoint** | `v1/chat/completions` |
| **Model** | `OPENAI_MODEL` trong `.env` (mặc định `gpt-5.5-2026-04-23`) |
| **API Key** | `OPENAI_API_KEY` trong `.env` |
| **Temperature** | `0.3` (chỉ với `gpt-4.x`; **bỏ** với `gpt-5.5`) |
| **max_tokens** | Không set |
| **response_format** | `{"type": "json_object"}` |
| **Batch size** | 15 segment / request |
| **Topic** | `TRANSLATION_TOPIC` → `economics` / `everyday` / `humor` |
| **Target language** | `vi` (hardcoded trong pipeline EN) |

---

## Kiểm tra model đang dùng

Khi web server chạy, gọi:

```
GET http://127.0.0.1:8000/api/defaults
```

Response:

```json
{
  "subtitle_font_size": 80,
  "subtitle_font_color": "#9333EA",
  "openai_model": "gpt-5.5-2026-04-23"
}
```

---

## Ghi chú

- Flow này chỉ dùng cho **dịch subtitle EN → VI**.
- Mode **VI** (PhoWhisper) **không dịch** — có flow OpenAI riêng sửa loanword trong `vi_loanword_fix.py` (prompt và mục đích khác).
- Console log khi dịch: `Translating segments 1-15...` (theo batch).

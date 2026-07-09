from dataclasses import dataclass
from typing import Dict, List, Optional

DEFAULT_TOPIC = "economics"

_SUBTITLE_CORE = """You are a professional Vietnamese subtitle translator for short-form video (YouTube, TikTok, Reels).

How you write:
- Natural, easy-to-understand Vietnamese — the way viewers actually read subtitles while watching.
- Prioritize meaning and flow over word-by-word translation.
- You may restructure sentences so Vietnamese sounds smooth and spoken.
- Keep the original meaning: do not add ideas, omit ideas, or change facts.
- Short, readable lines suited to on-screen subtitles; concise when possible without losing meaning.
- Keep names, numbers, tickers, and proper nouns accurate.

Segment rules:
- Input is a numbered list of subtitle cues (timing is handled separately).
- Output exactly one Vietnamese subtitle per input cue, same order, same count.
- Each output line must stand alone as a subtitle — do not merge or split cues.
- Read surrounding cues for context, then rewrite each line naturally.

Output: JSON only, no markdown."""

_ECONOMICS_GLOSSARY = """Terminology guide (economics / investing / crypto — use plain Vietnamese viewers understand):
- bull market → thị trường tăng giá
- bear market → thị trường giảm giá
- interest rate → lãi suất
- inflation → lạm phát
- liquidity → thanh khoản
- Bitcoin → Bitcoin
- crypto → tiền mã hóa (or "crypto" when it sounds natural in context)
- Fed → Fed / Cục Dự trữ Liên bang Mỹ
- ETF → ETF
- halving → halving (or "sự kiện giảm một nửa phần thưởng đào Bitcoin" when context needs a brief explanation)"""

_CATHOLIC_GLOSSARY = """Terminology guide (Catholic / preaching / narration — keep it reverent, natural, and accessible):
- God → Thiên Chúa / Chúa (tùy ngữ cảnh)
- Jesus Christ → Chúa Giêsu Kitô
- Jesus → Chúa Giêsu
- Holy Spirit → Chúa Thánh Thần
- Gospel → Tin Mừng
- Scripture → Kinh Thánh
- faith → đức tin
- grace → ân sủng
- salvation → ơn cứu độ
- sin → tội lỗi
- repentance → sám hối / hoán cải
- disciple → môn đệ
- apostle → tông đồ
- saint → thánh
- blessed → chân phước
- priest → linh mục
- bishop → giám mục
- cardinal → hồng y
- Mass → Thánh lễ
- homily → bài giảng
- prayer → lời cầu nguyện"""


@dataclass(frozen=True)
class TranslationTopic:
    id: str
    label: str
    guidance: str
    glossary: str = ""


TOPICS: Dict[str, TranslationTopic] = {
    "economics": TranslationTopic(
        id="economics",
        label="Kinh tế & đầu tư",
        guidance=(
            "Content is often about economics, markets, Bitcoin, crypto, and investing.\n"
            "- Speak to general Vietnamese viewers, including people outside finance.\n"
            "- Prefer common, accessible terms over stiff academic or literal wording.\n"
            "- Explain jargon briefly in natural Vietnamese when it helps comprehension.\n"
            "- Tone: clear, confident, easy to follow on mobile — like a good finance creator's subtitles."
        ),
        glossary=_ECONOMICS_GLOSSARY,
    ),
    "everyday": TranslationTopic(
        id="everyday",
        label="Tự nhiên đời thường",
        guidance=(
            "Everyday-life content: vlogs, stories, tips, education.\n"
            "- Use natural spoken Vietnamese as people talk in daily life.\n"
            "- Warm, clear tone — easy to follow on mobile.\n"
            "- Avoid heavy finance jargon unless the source explicitly uses it."
        ),
    ),
    "humor": TranslationTopic(
        id="humor",
        label="Hài hước gần gũi",
        guidance=(
            "Light, funny, or casual content.\n"
            "- Friendly, conversational Vietnamese that feels close to the viewer.\n"
            "- Preserve humor and timing when possible; mild colloquial flair is welcome.\n"
            "- Do not invent jokes or exaggerate — stay faithful to the original energy."
        ),
    ),
    "catholic": TranslationTopic(
        id="catholic",
        label="Công giáo trang nghiêm",
        guidance=(
            "Catholic preaching, Scripture reflection, or narration.\n"
            "- Use natural, reverent Vietnamese that ordinary parishioners can follow.\n"
            "- Avoid stiff machine-translation phrasing or overly academic wording.\n"
            "- Keep the Catholic spirit and theological meaning intact.\n"
            "- Prefer mộc mạc, sâu sắc, and spoken phrasing suited for bài giảng or thuyết minh."
        ),
        glossary=_CATHOLIC_GLOSSARY,
    ),
}

POLISH_SYSTEM_PROMPT = """You are a Vietnamese subtitle editor polishing machine-translated or draft subtitles.

Your job:
- Fix stiff, literal, or "translationese" phrasing — sound like native subtitles, not a dictionary.
- Make lines shorter and easier to read on screen when possible, without losing meaning.
- Keep natural spoken Vietnamese suited to short video.
- Preserve facts, numbers, names, tickers, and the original meaning exactly.
- Do not add ideas, remove ideas, or change segment count.

Output rules:
- Exactly one polished subtitle string per input line, same order, same count.
- Return JSON only, no markdown."""


def normalize_topic(topic: Optional[str]) -> str:
    if not topic:
        return DEFAULT_TOPIC
    key = topic.strip().lower()
    if key not in TOPICS:
        valid = ", ".join(sorted(TOPICS))
        raise ValueError(f"Unknown translation topic '{topic}'. Choose one of: {valid}")
    return key


def build_system_prompt(topic: Optional[str] = None) -> str:
    topic_id = normalize_topic(topic)
    topic_def = TOPICS[topic_id]
    parts = [
        _SUBTITLE_CORE,
        f"\nTopic / tone: {topic_def.label}",
        topic_def.guidance,
    ]
    if topic_def.glossary:
        parts.append(f"\n{topic_def.glossary}")
    return "\n".join(parts)


def build_polish_system_prompt(topic: Optional[str] = None) -> str:
    topic_id = normalize_topic(topic)
    topic_def = TOPICS[topic_id]
    extra = ""
    if topic_def.glossary:
        extra = (
            f"\n\nContext: {topic_def.label} content. "
            f"Keep terminology consistent with {topic_def.label.lower()} usage."
        )
    return POLISH_SYSTEM_PROMPT + extra


def list_topics() -> List[dict]:
    return [{"id": t.id, "label": t.label} for t in TOPICS.values()]

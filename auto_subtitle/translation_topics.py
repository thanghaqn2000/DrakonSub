from dataclasses import dataclass
from typing import Dict, List, Optional

DEFAULT_TOPIC = "economics"

_BASE_RULES = """Rules (strict):
- Translate each segment faithfully from the English source. Do not omit ideas, add ideas, or change meaning.
- One English segment → exactly one Vietnamese segment, same order.
- Do not merge or split segments.
- Keep names, numbers, and proper nouns accurate.
- Return JSON only, no markdown."""


@dataclass(frozen=True)
class TranslationTopic:
    id: str
    label: str
    guidance: str


TOPICS: Dict[str, TranslationTopic] = {
    "economics": TranslationTopic(
        id="economics",
        label="Kinh tế",
        guidance=(
            "Audience: general viewers, including people outside economics/finance. "
            "Content is often about economics.\n"
            "- Use natural, easy-to-understand Vietnamese. Slightly colloquial/friendly is fine.\n"
            "- Explain economics terms in plain language when needed; avoid stiff literal or academic wording."
        ),
    ),
    "everyday": TranslationTopic(
        id="everyday",
        label="Tự nhiên đời thường",
        guidance=(
            "Audience: general viewers watching everyday-life content (vlogs, stories, daily tips, education).\n"
            "- Use natural spoken Vietnamese as people talk in daily life.\n"
            "- Avoid economics jargon unless the source explicitly uses it; prefer plain, relatable wording.\n"
            "- Keep a warm, clear tone — easy to follow on mobile."
        ),
    ),
    "humor": TranslationTopic(
        id="humor",
        label="Hài hước gần gũi",
        guidance=(
            "Audience: general viewers watching light, funny, or casual content.\n"
            "- Use friendly, playful Vietnamese that feels close and conversational.\n"
            "- Preserve humor and timing when possible; mild colloquial flair is welcome.\n"
            "- Do not invent jokes or exaggerate — stay faithful to the original meaning and energy."
        ),
    ),
}


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
    return (
        "You translate English video subtitles into Vietnamese.\n\n"
        f"Topic / tone: {topic_def.label}\n"
        f"{topic_def.guidance}\n\n"
        f"{_BASE_RULES}"
    )


def list_topics() -> List[dict]:
    return [{"id": t.id, "label": t.label} for t in TOPICS.values()]

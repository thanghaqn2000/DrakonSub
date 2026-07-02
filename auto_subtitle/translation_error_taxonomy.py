"""General subtitle translation error taxonomy — reusable across videos and topics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ErrorTypeDefinition:
    id: str
    description: str
    common_symptoms: str
    likely_root_cause: str
    fix_strategy: str
    auto_repair: bool
    human_review: bool


ERROR_TYPES: Dict[str, ErrorTypeDefinition] = {
    "cue_fragmentation_error": ErrorTypeDefinition(
        id="cue_fragmentation_error",
        description="A single thought is split awkwardly across multiple subtitle cues.",
        common_symptoms="Cue ends mid-clause; next cue continues grammar; isolated words like conjunctions at cue end.",
        likely_root_cause="ASR/subtitle segmentation breaks on timing, not semantics.",
        fix_strategy="Read neighboring cues as one meaning unit; redistribute natural Vietnamese across original cue indexes.",
        auto_repair=True,
        human_review=False,
    ),
    "domain_term_error": ErrorTypeDefinition(
        id="domain_term_error",
        description="Domain-specific term translated incorrectly or too academically for general viewers.",
        common_symptoms="Wrong technical meaning; jargon without explanation; inconsistent term across cues.",
        likely_root_cause="Missing topic glossary or literal dictionary translation.",
        fix_strategy="Use video context key_terms; prefer plain Vietnamese explanations for beginners.",
        auto_repair=True,
        human_review=True,
    ),
    "split_term_across_cues_error": ErrorTypeDefinition(
        id="split_term_across_cues_error",
        description="A compound noun or technical phrase is broken across cue boundaries.",
        common_symptoms="Partial term at cue end/start; unreadable phrase when cues read separately.",
        likely_root_cause="Source segmentation splits inside a multi-word term.",
        fix_strategy="Group cues into meaning unit before translation; keep term integrity in distribution.",
        auto_repair=True,
        human_review=False,
    ),
    "pronoun_reference_error": ErrorTypeDefinition(
        id="pronoun_reference_error",
        description="Pronoun or demonstrative (it/this/that/they) resolved incorrectly in Vietnamese.",
        common_symptoms="Wrong subject/object; ambiguous 'nó/điều đó/cái đó'; meaning shifts between cues.",
        likely_root_cause="Cue translated without wider context.",
        fix_strategy="Use neighboring cues and meaning units to resolve reference; prefer explicit noun when needed.",
        auto_repair=True,
        human_review=False,
    ),
    "idiom_metaphor_error": ErrorTypeDefinition(
        id="idiom_metaphor_error",
        description="Idiom, metaphor, or figurative speech translated literally.",
        common_symptoms="Unnatural word-for-word phrasing; lost figurative meaning; confusing imagery.",
        likely_root_cause="Model translates surface words instead of intended meaning.",
        fix_strategy="Translate intended meaning into natural spoken Vietnamese; avoid calque.",
        auto_repair=True,
        human_review=True,
    ),
    "sarcasm_tone_error": ErrorTypeDefinition(
        id="sarcasm_tone_error",
        description="Speaker tone (sarcasm, irony, emphasis) not reflected appropriately.",
        common_symptoms="Flat translation; opposite tone; missing emphasis particles where natural.",
        likely_root_cause="Lack of tonal/contextual analysis.",
        fix_strategy="Use video tone summary; preserve speaker intent without over-acting.",
        auto_repair=False,
        human_review=True,
    ),
    "asr_possible_error": ErrorTypeDefinition(
        id="asr_possible_error",
        description="Source English may be a transcription error.",
        common_symptoms="Nonsense phrase; homophone confusion; unlikely word in context.",
        likely_root_cause="Speech recognition mistake.",
        fix_strategy="Flag for review; infer likely meaning from context; do not invent facts.",
        auto_repair=False,
        human_review=True,
    ),
    "literal_translation_error": ErrorTypeDefinition(
        id="literal_translation_error",
        description="Vietnamese follows English structure too closely and sounds unnatural.",
        common_symptoms="Stiff phrasing; wrong collocation; English word order in Vietnamese.",
        likely_root_cause="Cue-by-cue translation without meaning-unit rewrite.",
        fix_strategy="Rewrite for natural spoken Vietnamese while preserving meaning.",
        auto_repair=True,
        human_review=False,
    ),
    "over_compression_error": ErrorTypeDefinition(
        id="over_compression_error",
        description="Subtitle shortened so much that meaning is lost or distorted.",
        common_symptoms="Missing key idea; ambiguous short phrase; pronoun without antecedent.",
        likely_root_cause="Aggressive length reduction.",
        fix_strategy="Restore essential meaning; shorten only filler words.",
        auto_repair=True,
        human_review=True,
    ),
    "readability_cps_error": ErrorTypeDefinition(
        id="readability_cps_error",
        description="Too many characters per second for comfortable reading.",
        common_symptoms="Long text in short duration; viewer cannot finish reading.",
        likely_root_cause="Verbose translation or short cue timing.",
        fix_strategy="Shorten phrasing without losing core meaning; flag if timing cannot support text.",
        auto_repair=True,
        human_review=False,
    ),
    "unnatural_vietnamese_error": ErrorTypeDefinition(
        id="unnatural_vietnamese_error",
        description="Grammar or phrasing is not how Vietnamese speakers would say it.",
        common_symptoms="Awkward particles; wrong register; machine-translated feel.",
        likely_root_cause="Insufficient localization pass.",
        fix_strategy="Rewrite to simple, natural Vietnamese for general adult audience.",
        auto_repair=True,
        human_review=False,
    ),
    "missing_or_empty_cue_error": ErrorTypeDefinition(
        id="missing_or_empty_cue_error",
        description="A non-empty source cue has blank or missing translation.",
        common_symptoms="Empty subtitle line; dropped cue; count mismatch.",
        likely_root_cause="Model merged/skipped cue or validation gap.",
        fix_strategy="Always output one non-empty Vietnamese line per non-empty source cue.",
        auto_repair=True,
        human_review=False,
    ),
    "cue_flow_error": ErrorTypeDefinition(
        id="cue_flow_error",
        description="Adjacent cues do not read smoothly together in Vietnamese.",
        common_symptoms="Standalone fragments; weak cue starters; broken cross-cue sentence.",
        likely_root_cause="Independent cue editing without cross-cue coherence.",
        fix_strategy="Smooth flow across cues while preserving cue count and timestamps.",
        auto_repair=True,
        human_review=False,
    ),
    "semantic_drift_error": ErrorTypeDefinition(
        id="semantic_drift_error",
        description="Vietnamese meaning diverges from English source intent.",
        common_symptoms="Added ideas; removed key idea; wrong logical relation.",
        likely_root_cause="Over-rewrite or context misunderstanding.",
        fix_strategy="Re-align to source meaning; compare unit-level semantics.",
        auto_repair=True,
        human_review=True,
    ),
    "semantic_alignment_error": ErrorTypeDefinition(
        id="semantic_alignment_error",
        description="Vietnamese cue text appears aligned to a different English cue or unit.",
        common_symptoms="VI belongs to later/earlier idea; concepts from wrong source cue; shifted repair output.",
        likely_root_cause="Cue-by-cue repair/translation without index discipline.",
        fix_strategy="Re-map VI to matching source cue by meaning; preserve cue order within unit.",
        auto_repair=True,
        human_review=True,
    ),
    "repair_contract_violation": ErrorTypeDefinition(
        id="repair_contract_violation",
        description="Model repair output broke the repair contract (empty cue, wrong indexes, missing cues).",
        common_symptoms="Empty repaired text; wrong unit_id; missing cue; idea moved to wrong index.",
        likely_root_cause="Repair model ignored structure constraints.",
        fix_strategy="Reject invalid repair; keep original VI; flag for human review.",
        auto_repair=False,
        human_review=True,
    ),
    "cue_alignment_warning": ErrorTypeDefinition(
        id="cue_alignment_warning",
        description="Cue alone looks weak or generic but the full meaning unit is accurate and ordered.",
        common_symptoms="Fragment cue; phrase split across unit; low per-cue overlap but unit OK.",
        likely_root_cause="Multi-cue English phrase distributed across subtitle cues.",
        fix_strategy="Optional polish only; do not treat as severe drift if unit semantics match.",
        auto_repair=False,
        human_review=False,
    ),
    "repeated_meaning_error": ErrorTypeDefinition(
        id="repeated_meaning_error",
        description="Adjacent cues repeat the same Vietnamese idea without source justification.",
        common_symptoms="Same question twice; redundant phrase in cue N and N+1.",
        likely_root_cause="Repair or translation duplicated content across cues.",
        fix_strategy="Rewrite the whole meaning unit; distribute each idea once across cues.",
        auto_repair=True,
        human_review=False,
    ),
    "possible_asr_term_unresolved": ErrorTypeDefinition(
        id="possible_asr_term_unresolved",
        description="A flagged ASR-risk source phrase was translated literally and still sounds unnatural.",
        common_symptoms="Odd proper noun; homophone phrase; unlikely term kept verbatim in VI.",
        likely_root_cause="Uncertain speech recognition; translator copied surface form.",
        fix_strategy="Review context; infer likely intent only if confident; otherwise flag for human.",
        auto_repair=False,
        human_review=True,
    ),
}


def get_error_type(error_id: str) -> ErrorTypeDefinition:
    if error_id not in ERROR_TYPES:
        raise KeyError(f"Unknown error type: {error_id}")
    return ERROR_TYPES[error_id]


def list_error_types() -> List[ErrorTypeDefinition]:
    return list(ERROR_TYPES.values())


def auto_repair_error_ids() -> List[str]:
    return [e.id for e in ERROR_TYPES.values() if e.auto_repair]


def human_review_error_ids() -> List[str]:
    return [e.id for e in ERROR_TYPES.values() if e.human_review]

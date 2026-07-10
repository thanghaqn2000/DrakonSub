import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.voiceover.srt_quality import (  # noqa: E402
    compact_voiceover_cues,
    compute_voiceover_srt_metrics,
    group_source_cues_for_voiceover,
    optimize_voiceover_srt_entries,
    reindex_drop_empty,
    voiceover_narration_translation_context,
)


def _cue(start: str, end: str, text: str) -> dict:
    return {"start_str": start, "end_str": end, "text": text}


class ReindexDropEmptyTests(unittest.TestCase):
    def test_drops_whitespace_only_and_reindexes(self) -> None:
        entries = [
            _cue("00:00:00,000", "00:00:01,000", "Một"),
            _cue("00:00:01,000", "00:00:01,500", " "),
            _cue("00:00:01,500", "00:00:02,500", "Hai"),
            _cue("00:00:02,500", "00:00:03,000", ""),
        ]
        out = reindex_drop_empty(entries)
        self.assertEqual(len(out), 2)
        self.assertEqual([e["text"] for e in out], ["Một", "Hai"])

    def test_metrics_report_continuous_indexes(self) -> None:
        entries = [
            _cue("00:00:00,000", "00:00:02,000", "Một câu đủ dài."),
            _cue("00:00:02,200", "00:00:04,200", "Hai câu đủ dài."),
        ]
        metrics = compute_voiceover_srt_metrics(entries)
        self.assertEqual(metrics["cue_count"], 2)
        self.assertEqual(metrics["max_index"], 2)
        self.assertEqual(metrics["missing_indexes"], [])


class GroupSourceCuesTests(unittest.TestCase):
    def test_merges_mid_phrase_fragments(self) -> None:
        # Mirrors SA example: "I would not" / "believe it."
        entries = [
            _cue("00:00:00,000", "00:00:01,500", "If I told you at twenty"),
            _cue("00:00:01,600", "00:00:03,000", "that life is short and"),
            _cue("00:00:03,100", "00:00:04,200", "I would not"),
            _cue("00:00:04,300", "00:00:04,600", "believe it."),
        ]
        grouped = group_source_cues_for_voiceover(entries)
        self.assertLess(len(grouped), len(entries))
        joined = " ".join(e["text"] for e in grouped)
        self.assertIn("believe it", joined.lower())
        # Should not leave a tiny tail cue alone when mergeable
        short_tails = [e for e in grouped if len(e["text"].split()) <= 2]
        self.assertEqual(short_tails, [])

    def test_keeps_sentence_boundary_separate(self) -> None:
        entries = [
            _cue("00:00:00,000", "00:00:02,000", "Life is short."),
            _cue("00:00:02,500", "00:00:04,500", "What will you do?"),
        ]
        grouped = group_source_cues_for_voiceover(entries)
        self.assertEqual(len(grouped), 2)


class CompactVoiceoverCuesTests(unittest.TestCase):
    def test_merges_short_tail_cues(self) -> None:
        entries = [
            _cue("00:00:00,000", "00:00:02,000", "Và nếu tôi nói với bạn điều đó, bạn cũng không"),
            _cue("00:00:02,100", "00:00:02,340", "tin đâu."),
        ]
        compacted = compact_voiceover_cues(entries)
        self.assertEqual(len(compacted), 1)
        self.assertIn("tin đâu", compacted[0]["text"])

    def test_expands_timing_when_cps_too_high(self) -> None:
        # Long text in short window, with gap after to borrow
        entries = [
            _cue(
                "00:00:00,000",
                "00:00:01,000",
                "Cuộc đời Ngài kết thúc trên thập tự giá vì chúng ta.",
            ),
            _cue("00:00:03,000", "00:00:05,000", "Hãy nhớ điều đó."),
        ]
        compacted = compact_voiceover_cues(entries, target_cps=18.0)
        first = compacted[0]
        # Should borrow into the gap before next cue
        self.assertNotEqual(first["end_str"], "00:00:01,000")


class OptimizePipelineTests(unittest.TestCase):
    def test_full_optimize_drops_empty_and_reindexes(self) -> None:
        entries = [
            _cue("00:00:00,000", "00:00:01,800", "Thời gian thì ngắn lắm"),
            _cue("00:00:01,900", "00:00:02,100", " "),
            _cue("00:00:02,200", "00:00:02,500", "tin đâu."),
            _cue("00:00:03,000", "00:00:05,000", "Hãy sống có ý nghĩa."),
        ]
        out = optimize_voiceover_srt_entries(entries)
        self.assertTrue(all(e["text"].strip() for e in out))
        metrics = compute_voiceover_srt_metrics(out)
        self.assertEqual(metrics["missing_indexes"], [])
        self.assertEqual(metrics["cue_count"], metrics["max_index"])
        self.assertLessEqual(len(out), 3)

    def test_narration_context_mentions_tts(self) -> None:
        ctx = voiceover_narration_translation_context()
        style = ctx["video_context"]["translation_style"].lower()
        self.assertIn("tts", style)
        self.assertIn("narration", style)


if __name__ == "__main__":
    unittest.main()

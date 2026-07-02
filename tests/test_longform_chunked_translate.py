import unittest

from auto_subtitle.raw_llm_response_cache import build_cache_key


class LongformChunkedTranslateTests(unittest.TestCase):
    def test_cache_key_changes_when_chunk_scope_changes(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ]
        key_a = build_cache_key(
            llm_task_type="longform_chunk_translate",
            model="gpt-4o-mini",
            messages=messages,
            batch_indices=[1, 2, 3],
            cache_scope={"chunk_id": "chunk_001", "chunk_cue_indices": [1, 2, 3]},
        )
        key_b = build_cache_key(
            llm_task_type="longform_chunk_translate",
            model="gpt-4o-mini",
            messages=messages,
            batch_indices=[1, 2, 3],
            cache_scope={"chunk_id": "chunk_002", "chunk_cue_indices": [4, 5, 6]},
        )
        self.assertNotEqual(key_a, key_b)

    def test_build_longform_chunks_prefers_sentence_boundaries(self) -> None:
        from auto_subtitle.longform_chunked_translate import build_longform_chunks

        entries = [
            {"text": "First thought"},
            {"text": "still same thought"},
            {"text": "ends here."},
            {"text": "New idea"},
            {"text": "continues"},
            {"text": "stops now."},
        ]

        chunks = build_longform_chunks(
            entries,
            target_chunk_size=3,
            min_chunk_size=2,
            max_chunk_size=4,
        )

        self.assertEqual([chunk.entry_indexes for chunk in chunks], [[0, 1, 2], [3, 4, 5]])

    def test_parse_longform_response_requires_exact_cue_indexes(self) -> None:
        from auto_subtitle.longform_chunked_translate import parse_longform_chunk_response

        content = """
        {
          "translations": [
            {"cue_index": 1, "vi": "xin chao"},
            {"cue_index": 3, "vi": "tam biet"}
          ],
          "chunk_summary_en": "summary",
          "chunk_summary_vi": "tom tat"
        }
        """

        with self.assertRaises(ValueError):
            parse_longform_chunk_response(content, expected_cue_indexes=[1, 2])

    def test_parse_longform_response_uses_source_fallback_for_empty_vi(self) -> None:
        from auto_subtitle.longform_chunked_translate import parse_longform_chunk_response

        content = """
        {
          "translations": [
            {"cue_index": 38, "vi": ""}
          ],
          "chunk_summary_en": "",
          "chunk_summary_vi": ""
        }
        """

        parsed = parse_longform_chunk_response(
            content,
            expected_cue_indexes=[38],
            fallback_by_index={38: "source fallback"},
        )

        self.assertEqual(parsed["translations"][38], "source fallback")


if __name__ == "__main__":
    unittest.main()

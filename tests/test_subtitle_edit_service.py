import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from auto_subtitle import web


SOURCE_SRT = """1
00:00:00,000 --> 00:00:02,000
Outsiders
change the world.

2
00:00:02,000 --> 00:00:04,000
Stay curious.
"""

VI_FINAL_SRT = """1
00:00:00,000 --> 00:00:02,000
Những kẻ ngoài cuộc
thay đổi thế giới.

2
00:00:02,000 --> 00:00:04,000
Hãy giữ sự tò mò.
"""


class SubtitleEditServiceTests(unittest.TestCase):
    def test_load_write_and_effective_path(self) -> None:
        from auto_subtitle.subtitle_edit_service import (
            SubtitleCue,
            get_effective_vi_srt,
            load_srt,
            write_srt,
        )

        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            vi_final = job_dir / "vi_final.srt"
            vi_final.write_text(VI_FINAL_SRT, encoding="utf-8")

            cues = load_srt(vi_final)
            self.assertEqual(cues[0].text, "Những kẻ ngoài cuộc\nthay đổi thế giới.")

            edited = [
                SubtitleCue(index=c.index, start=c.start, end=c.end, text=c.text)
                for c in cues
            ]
            edited[0].text = "Những người ngoài cuộc\nthay đổi thế giới."
            edited_path = job_dir / "edited_vi.srt"
            write_srt(edited, edited_path)

            self.assertEqual(get_effective_vi_srt(job_dir), edited_path)

    def test_merge_apply_and_reset_behavior(self) -> None:
        from auto_subtitle.subtitle_edit_service import (
            apply_text_edits,
            load_srt,
            merge_subtitle_views,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "source.srt"
            vi_final = tmpdir / "vi_final.srt"
            source.write_text(SOURCE_SRT, encoding="utf-8")
            vi_final.write_text(VI_FINAL_SRT, encoding="utf-8")

            source_cues = load_srt(source)
            original = load_srt(vi_final)
            current = load_srt(vi_final)

            current = apply_text_edits(
                original,
                current,
                [{"cue_index": 1, "text": "Những người ngoài cuộc\nthay đổi thế giới."}],
            )
            merged = merge_subtitle_views(source_cues, original, current)

            self.assertEqual(merged[0]["source_text"], "Outsiders\nchange the world.")
            self.assertTrue(merged[0]["edited"])
            self.assertEqual(current[0].start, original[0].start)
            self.assertEqual(len(current), len(original))


class SubtitleEditApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.jobs_root = Path(self.tmp.name)
        self.original_root = web.JOBS_ROOT
        self.original_jobs = web.jobs
        self.client = TestClient(web.app)
        web.JOBS_ROOT = self.jobs_root
        web.jobs = {}

        job_dir = self.jobs_root / "job123"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "source.srt").write_text(SOURCE_SRT, encoding="utf-8")
        (job_dir / "vi_final.srt").write_text(VI_FINAL_SRT, encoding="utf-8")
        (job_dir / "input.mp4").write_bytes(b"fake-mp4")
        (job_dir / "layout.json").write_text("{}", encoding="utf-8")
        web.jobs["job123"] = web.Job(
            id="job123",
            status=web.JobStatus.DONE,
            output_name="demo",
            input_path=str(job_dir / "input.mp4"),
            source_srt_path=str(job_dir / "source.srt"),
            srt_path=str(job_dir / "vi_final.srt"),
            layout_path=str(job_dir / "layout.json"),
            layout_saved=True,
        )

    def tearDown(self) -> None:
        web.JOBS_ROOT = self.original_root
        web.jobs = self.original_jobs
        self.tmp.cleanup()

    def test_get_save_reset_and_reset_all(self) -> None:
        response = self.client.get("/api/jobs/job123/subtitles")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["has_edits"])

        save_response = self.client.post(
            "/api/jobs/job123/subtitles/save",
            json={"edits": [{"cue_index": 1, "text": "Những người ngoài cuộc thay đổi thế giới."}]},
        )
        self.assertEqual(save_response.status_code, 200)

        edited_path = self.jobs_root / "job123" / "edited_vi.srt"
        self.assertTrue(edited_path.exists())

        get_after_save = self.client.get("/api/jobs/job123/subtitles")
        payload = get_after_save.json()
        self.assertTrue(payload["has_edits"])
        self.assertEqual(payload["subtitle_source"], "edited_vi.srt")
        self.assertTrue(payload["subtitles"][0]["edited"])

        reset_one = self.client.post("/api/jobs/job123/subtitles/reset", json={"cue_indices": [1]})
        self.assertEqual(reset_one.status_code, 200)
        self.assertFalse(self.client.get("/api/jobs/job123/subtitles").json()["has_edits"])

        self.client.post(
            "/api/jobs/job123/subtitles/save",
            json={"edits": [{"cue_index": 2, "text": "Giữ lấy sự tò mò."}]},
        )
        reset_all = self.client.post("/api/jobs/job123/subtitles/reset-all")
        self.assertEqual(reset_all.status_code, 200)
        self.assertFalse((self.jobs_root / "job123" / "edited_vi.srt").exists())
        edits_json = json.loads((self.jobs_root / "job123" / "user_edits.json").read_text(encoding="utf-8"))
        self.assertEqual(edits_json["edits"], [])

    def test_render_reports_current_subtitle_source(self) -> None:
        response = self.client.post("/api/jobs/job123/render")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["subtitle_source"], "vi_final.srt")

    def test_reburn_uses_edited_vi_srt_when_present(self) -> None:
        original_vi_text = (self.jobs_root / "job123" / "vi_final.srt").read_text(encoding="utf-8")
        save_response = self.client.post(
            "/api/jobs/job123/subtitles/save",
            json={"edits": [{"cue_index": 2, "text": "TEST_SUB_EDIT_123."}]},
        )
        self.assertEqual(save_response.status_code, 200)
        edited_path = self.jobs_root / "job123" / "edited_vi.srt"
        self.assertIn("TEST_SUB_EDIT_123.", edited_path.read_text(encoding="utf-8"))
        captured = {}

        def fake_reburn(video_path, srt_path, output_path, layout, config=None, on_progress=None):
            captured["video_path"] = video_path
            captured["srt_path"] = srt_path
            captured["output_path"] = output_path
            Path(output_path).write_bytes(b"reburned")
            return output_path

        with patch.object(web, "reburn_subtitles", side_effect=fake_reburn):
            output_path = web.reburn_job_subtitles("job123")

        self.assertEqual(Path(captured["srt_path"]).name, "edited_vi.srt")
        self.assertEqual(web.jobs["job123"].last_render_subtitle_source, "edited_vi.srt")
        self.assertTrue(Path(output_path).exists())
        self.assertEqual(
            (self.jobs_root / "job123" / "vi_final.srt").read_text(encoding="utf-8"),
            original_vi_text,
        )

    def test_reburn_falls_back_to_vi_final_without_edits(self) -> None:
        captured = {}

        def fake_reburn(video_path, srt_path, output_path, layout, config=None, on_progress=None):
            captured["srt_path"] = srt_path
            Path(output_path).write_bytes(b"reburned")
            return output_path

        with patch.object(web, "reburn_subtitles", side_effect=fake_reburn):
            web.reburn_job_subtitles("job123")

        self.assertEqual(Path(captured["srt_path"]).name, "vi_final.srt")
        self.assertEqual(web.jobs["job123"].last_render_subtitle_source, "vi_final.srt")


if __name__ == "__main__":
    unittest.main()

import os
import unittest

from auto_subtitle.subtitle_renderer import (
    FONT_PRESETS,
    SubtitleRenderStyle,
    default_layout_dict,
    resolve_background_visible,
)
from auto_subtitle.web import validate_layout


class SubtitleStyleControlsTests(unittest.TestCase):
    def test_default_layout_solid_background_and_visible(self):
        layout = default_layout_dict()
        self.assertEqual(layout["background_opacity"], 1.0)
        self.assertTrue(layout["background_visible"])

    def test_resolve_background_visible_missing_defaults_on(self):
        self.assertTrue(resolve_background_visible({}))
        self.assertTrue(resolve_background_visible(None))

    def test_resolve_background_visible_false(self):
        self.assertFalse(resolve_background_visible({"background_visible": False}))
        self.assertFalse(resolve_background_visible({"background_visible": "false"}))

    def test_validate_layout_accepts_background_visible_and_new_fonts(self):
        layout = validate_layout(
            {
                "background_visible": False,
                "font_family": "comfortaa",
                "background_opacity": 1.0,
            }
        )
        self.assertFalse(layout["background_visible"])
        self.assertEqual(layout["font_family"], "comfortaa")

    def test_validate_layout_old_job_without_background_visible(self):
        layout = validate_layout({"font_family": "arial_bold"})
        self.assertTrue(layout["background_visible"])

    def test_local_font_files_exist(self):
        for key in ("comfortaa", "montserrat_alternates"):
            path = FONT_PRESETS[key][0]
            self.assertTrue(os.path.isfile(path), f"Missing font file for {key}: {path}")

    def test_style_from_dict_background_visible(self):
        style = SubtitleRenderStyle.from_dict({"background_visible": False})
        self.assertFalse(style.background_visible)
        style_default = SubtitleRenderStyle.from_dict({})
        self.assertTrue(style_default.background_visible)


if __name__ == "__main__":
    unittest.main()

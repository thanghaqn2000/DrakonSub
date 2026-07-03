"""
Subtitle Renderer — Rounded Modern Style
=========================================
Renders Vietnamese subtitles with a soft rounded-rectangle background using
Pillow and overlays them onto the video with FFmpeg.

Why not ASS/libass?
  ASS BorderStyle=3 produces flat rectangular boxes — no border-radius support
  exists in the spec.  Pillow gives us full control over shape, opacity, and
  typography, then FFmpeg composites each cue image at the right time.

Render pipeline (rounded mode):
  1. Probe video size via ffmpeg.
  2. For each subtitle cue, render a transparent RGBA PNG:
       • rounded_rectangle background (Pillow ≥ 8.2)
       • word-wrapped, centered Vietnamese text
  3. Build an FFmpeg filter_complex that overlays each PNG with a time gate:
       overlay=x:y:enable='between(t,start,end)'
  4. Long filter strings (>100 kB) are written to a temp script file to avoid
     shell argument length limits.
  5. On any failure, fall back transparently to the existing ASS-based burn.

Classic mode (fallback or explicit):
  Uses the existing prepare_burn_subtitles() + libass pipeline unchanged.
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_FONTS_DIR = Path(__file__).resolve().parent / "fonts"

# ---------------------------------------------------------------------------
# Optional Pillow import
# ---------------------------------------------------------------------------

try:
    from PIL import Image, ImageDraw, ImageFont

    _PILLOW_OK = True
    # rounded_rectangle was added in Pillow 8.2.0
    _ROUNDED_RECT_OK = callable(getattr(ImageDraw.ImageDraw, "rounded_rectangle", None))
except ImportError:
    _PILLOW_OK = False
    _ROUNDED_RECT_OK = False


# ---------------------------------------------------------------------------
# Style config
# ---------------------------------------------------------------------------

@dataclass
class SubtitleRenderStyle:
    """
    All visual parameters for the subtitle renderer.

    Defaults match the spec: rounded modern look, purple text, white background.
    All pixel/ratio values are scaled responsively at render time.
    """

    mode: str = "rounded"                  # "rounded" | "classic"
    border_radius: int = 18               # px at reference_height
    padding_x: int = 28                   # horizontal inner padding (px at ref)
    padding_y: int = 16                   # vertical inner padding (px at ref)
    text_safe_padding_y: int = 12         # extra top/bottom room for diacritics/descenders (px at ref)
    background_color: str = "#FFFFFF"
    background_opacity: float = 1.0       # 0-1; default solid (no transparency)
    background_visible: bool = True         # when False, render text only (no box)
    text_color: str = "#9333EA"           # default purple; override via env
    font_size: int = 55                   # px at reference_height
    bottom_margin_ratio: float = 0.11     # fraction of video height
    max_width_ratio: float = 0.86         # fraction of video width
    line_spacing: float = 1.2
    reference_height: int = 1920          # used for responsive scaling
    x_ratio: Optional[float] = None       # box center X (0–1); None = bottom-center legacy
    y_ratio: Optional[float] = None       # box center Y (0–1); None = bottom-center legacy
    font_family: str = "arial_bold"      # preset key; see FONT_PRESETS
    font_bold: bool = True                # bold weight for subtitle text

    def to_dict(self) -> Dict[str, Any]:
        """Serialize style to a JSON-friendly layout dict."""
        data = asdict(self)
        data["width_ratio"] = self.max_width_ratio
        for key in ("text_safe_padding_y", "bottom_margin_ratio", "line_spacing", "reference_height"):
            data.pop(key, None)
        if self.x_ratio is None:
            data.pop("x_ratio", None)
        if self.y_ratio is None:
            data.pop("y_ratio", None)
        data.pop("max_width_ratio", None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubtitleRenderStyle":
        """Build a style from a layout JSON object."""
        known = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key == "width_ratio":
                kwargs["max_width_ratio"] = float(value)
            elif key in known:
                kwargs[key] = value
        if "max_width_ratio" not in kwargs and "max_width_ratio" in data:
            kwargs["max_width_ratio"] = float(data["max_width_ratio"])
        if "font_bold" not in kwargs:
            kwargs["font_bold"] = resolve_font_bold(data)
        return cls(**{k: v for k, v in kwargs.items() if k in known})

    def uses_custom_position(self) -> bool:
        """True when x_ratio and y_ratio drive placement instead of bottom margin."""
        return self.x_ratio is not None and self.y_ratio is not None


def default_layout_dict() -> Dict[str, Any]:
    """Default layout JSON for the layout editor (env values + editor defaults)."""
    style = load_render_style()
    return {
        "mode": style.mode,
        "x_ratio": 0.5,
        "y_ratio": 0.78,
        "width_ratio": style.max_width_ratio,
        "font_size": style.font_size,
        "text_color": style.text_color,
        "background_color": style.background_color,
        "background_opacity": style.background_opacity,
        "background_visible": style.background_visible,
        "border_radius": style.border_radius,
        "padding_x": style.padding_x,
        "padding_y": style.padding_y,
        "font_family": style.font_family,
        "font_bold": style.font_bold,
    }


def layout_dict_from_config(config) -> Dict[str, Any]:
    """Build initial layout JSON from a SubtitleConfig used during Generate."""
    layout = default_layout_dict()
    layout["font_size"] = config.subtitle_font_size
    layout["text_color"] = config.subtitle_font_color
    layout["background_color"] = config.subtitle_background_color
    layout["background_opacity"] = config.subtitle_background_opacity
    layout["border_radius"] = config.subtitle_border_radius
    layout["padding_x"] = config.subtitle_padding_x
    layout["padding_y"] = config.subtitle_padding_y
    layout["width_ratio"] = config.subtitle_max_width_ratio
    layout["mode"] = config.subtitle_style_mode
    if config.subtitle_bottom_margin_ratio is not None:
        layout["y_ratio"] = max(0.0, min(1.0, 1.0 - config.subtitle_bottom_margin_ratio))
    return layout


def load_render_style() -> SubtitleRenderStyle:
    """
    Build a SubtitleRenderStyle from environment variables.

    All fields fall back to conservative defaults when absent.
    """
    from .config import load_env
    load_env()

    def _s(k: str, d: str) -> str:
        return (os.getenv(k) or d).strip()

    def _i(k: str, d: int) -> int:
        try:
            return int(os.getenv(k) or d)
        except (ValueError, TypeError):
            return d

    def _f(k: str, d: float) -> float:
        try:
            return float(os.getenv(k) or d)
        except (ValueError, TypeError):
            return d

    return SubtitleRenderStyle(
        mode=_s("SUBTITLE_STYLE_MODE", "rounded"),
        border_radius=_i("SUBTITLE_BORDER_RADIUS", 18),
        padding_x=_i("SUBTITLE_PADDING_X", 28),
        padding_y=_i("SUBTITLE_PADDING_Y", 16),
        text_safe_padding_y=_i("SUBTITLE_TEXT_SAFE_PADDING_Y", 12),
        background_color=_s("SUBTITLE_BACKGROUND_COLOR", "#FFFFFF"),
        background_opacity=_f("SUBTITLE_BACKGROUND_OPACITY", 1.0),
        text_color=_s("SUBTITLE_TEXT_COLOR", _s("SUBTITLE_FONT_COLOR", "#9333EA")),
        font_size=_i("SUBTITLE_FONT_SIZE", 55),
        bottom_margin_ratio=_f("SUBTITLE_BOTTOM_MARGIN_RATIO", 0.11),
        max_width_ratio=_f("SUBTITLE_MAX_WIDTH_RATIO", 0.86),
        line_spacing=_f("SUBTITLE_LINE_SPACING", 1.2),
        reference_height=_i("SUBTITLE_REFERENCE_HEIGHT", 1920),
    )


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _hex_to_rgba(color: str, opacity: float) -> Tuple[int, int, int, int]:
    """Convert '#RRGGBB' + opacity (0-1) to an RGBA tuple."""
    h = color.strip().lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return r, g, b, int(opacity * 255)


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    h = color.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

# Font presets for layout editor (no arbitrary upload).
FONT_PRESETS: Dict[str, List[str]] = {
    "arial_bold": [
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
    "arial": [
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    "dejavu_bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
    "dejavu": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    "system_default": [],
    "comfortaa": [
        str(_FONTS_DIR / "Comfortaa-Regular.ttf"),
    ],
    "montserrat_alternates": [
        str(_FONTS_DIR / "MontserratAlternates-Regular.ttf"),
    ],
}

FONT_FAMILY_CHOICES = tuple(FONT_PRESETS.keys())

ASS_FONT_NAMES = {
    "arial_bold": "Arial",
    "arial": "Arial",
    "dejavu_bold": "DejaVu Sans",
    "dejavu": "DejaVu Sans",
    "system_default": "Arial",
    "comfortaa": "Comfortaa",
    "montserrat_alternates": "Montserrat Alternates",
}


def resolve_background_visible(layout: Optional[Dict[str, Any]] = None) -> bool:
    """True when subtitle background should be drawn; missing key => ON (backward compat)."""
    if not layout:
        return True
    value = layout.get("background_visible", True)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def resolve_font_bold(
    layout: Optional[Dict[str, Any]] = None,
    font_family: Optional[str] = None,
) -> bool:
    """True when subtitle text should use bold weight; missing key => infer from font_family."""
    family = (font_family or (layout or {}).get("font_family") or "arial_bold").strip().lower()
    if layout and "font_bold" in layout:
        value = layout["font_bold"]
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    return family.endswith("_bold")


def _font_preset_key(font_family: str, font_bold: bool) -> str:
    """Map layout font + bold toggle to a FONT_PRESETS key."""
    base = (font_family or "arial_bold").strip().lower()
    stem = base[:-5] if base.endswith("_bold") else base
    if stem == "system_default":
        return "system_default"
    if font_bold and stem in ("arial", "dejavu"):
        return f"{stem}_bold"
    if not font_bold and stem in ("arial", "dejavu"):
        return stem
    return stem if stem in FONT_PRESETS else "arial_bold"

# Ultimate fallback paths when preset candidates are exhausted.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _font_paths_for_preset(font_family: str, font_bold: bool = True) -> List[str]:
    """Ordered font file candidates for a named preset."""
    preset = _font_preset_key(font_family, font_bold)
    if preset == "system_default":
        return list(_FONT_CANDIDATES)
    paths = list(FONT_PRESETS.get(preset, []))
    if not paths:
        return list(_FONT_CANDIDATES)
    # De-duplicate while preserving order; append global fallbacks last.
    seen = set()
    ordered: List[str] = []
    for path in paths + _FONT_CANDIDATES:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def _load_font(
    size: int,
    font_family: str = "arial_bold",
    font_bold: bool = True,
) -> Tuple[Any, Optional[str]]:
    """
    Load a TrueType font at *size* points from a named preset.

    Returns (font, resolved_path). resolved_path is None only when falling back
    to PIL's built-in bitmap font.
    """
    for path in _font_paths_for_preset(font_family, font_bold):
        if os.path.isfile(path):
            try:
                font = ImageFont.truetype(path, size)
                return font, path
            except (OSError, ValueError) as exc:
                print(
                    f"WARNING: Failed to load font '{path}' at size {size}: {exc}"
                )
                continue
    print(
        "WARNING: No TrueType font found. Subtitle render may look incorrect."
    )
    return ImageFont.load_default(), None


# ---------------------------------------------------------------------------
# Text measurement & wrapping
# ---------------------------------------------------------------------------

def _measure_text(text: str, font) -> Tuple[int, int]:
    """Return (width, height) of *text* with *font*."""
    if hasattr(font, "getbbox"):
        bb = font.getbbox(text)
        return bb[2] - bb[0], bb[3] - bb[1]
    # Old Pillow / bitmap font fallback
    w, h = font.getsize(text) if hasattr(font, "getsize") else (len(text) * 8, 14)
    return w, h


def _font_line_height(font, fallback: int) -> int:
    """
    Robust per-line height covering full ascent + descent.

    Tight glyph bounding boxes vary per line and drop room for Vietnamese
    diacritics (above) and descenders like g/y (below).  Font metrics give a
    consistent slot so text never clips top/bottom.
    """
    try:
        ascent, descent = font.getmetrics()
        metric_h = ascent + descent
        if metric_h > 0:
            return metric_h
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        print(f"WARNING: Failed to read font metrics, using fallback height: {exc}")
        pass
    return fallback


def _wrap_text(text: str, font, max_width_px: int) -> List[str]:
    """
    Word-wrap *text* so each line fits within *max_width_px* pixels.

    Prefers short line count (≤ 2 lines for subtitle readability).
    Never breaks a word mid-token.
    """
    words = text.split()
    lines: List[str] = []
    current: List[str] = []

    for word in words:
        trial = " ".join(current + [word])
        w, _ = _measure_text(trial, font)
        if w <= max_width_px or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    return lines or [text]


# ---------------------------------------------------------------------------
# Per-cue overlay data
# ---------------------------------------------------------------------------

@dataclass
class _CueOverlay:
    """Metadata for one subtitle PNG overlay."""

    image_path: str
    x: int          # overlay x position in video
    y: int          # overlay y position in video
    start: float    # cue start in seconds
    end: float      # cue end in seconds


# ---------------------------------------------------------------------------
# Timestamp parser
# ---------------------------------------------------------------------------

def _parse_ts(ts: str) -> float:
    """Convert SRT timestamp ``HH:MM:SS,mmm`` to seconds."""
    ts = ts.strip()
    time_part, millis = ts.split(",")
    h, m, s = time_part.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(millis) / 1000.0


# ---------------------------------------------------------------------------
# Cue image renderer
# ---------------------------------------------------------------------------

def _render_cue_image(
    text: str,
    style: SubtitleRenderStyle,
    video_w: int,
    video_h: int,
    tmp_dir: str,
    index: int,
) -> Optional[_CueOverlay]:
    """
    Render one subtitle cue as a transparent RGBA PNG.

    Scales all metrics responsively from *style.reference_height* to the
    actual *video_h*.  Returns ``None`` for empty text.
    """
    text = text.strip()
    if not text:
        return None

    scale = video_h / max(style.reference_height, 1)
    max_width_ratio = min(1.0, max(0.1, style.max_width_ratio))
    line_spacing = max(1.0, style.line_spacing)
    background_opacity = min(1.0, max(0.0, style.background_opacity))
    bottom_margin_ratio = min(0.9, max(0.0, style.bottom_margin_ratio))
    font_size = max(12, round(style.font_size * scale))
    pad_x = max(6, round(style.padding_x * scale))
    pad_y = max(4, round(style.padding_y * scale))
    safe_pad_y = max(0, round(style.text_safe_padding_y * scale))
    radius = max(4, round(style.border_radius * scale))
    bottom_margin = round(video_h * bottom_margin_ratio)
    max_box_w = max(1, round(video_w * max_width_ratio))

    font, _font_path = _load_font(font_size, style.font_family, style.font_bold)

    # Word-wrap to fit inside the box (minus padding).
    inner_max_w = max(1, max_box_w - 2 * pad_x)
    lines = _wrap_text(text, font, inner_max_w)

    # Measure lines. Use font metrics for a stable line height that always has
    # room for diacritics (top) and descenders (bottom); also respect any tight
    # bbox that happens to be taller for a given line.
    line_widths = [_measure_text(ln, font)[0] for ln in lines]
    tight_heights = [_measure_text(ln, font)[1] for ln in lines]
    metric_line_h = _font_line_height(font, font_size)
    line_h = max([metric_line_h] + tight_heights) if tight_heights else metric_line_h
    line_gap = round(line_h * (line_spacing - 1.0))

    text_w = max(line_widths) if line_widths else inner_max_w
    text_total_h = len(lines) * line_h + max(0, len(lines) - 1) * line_gap

    box_w = min(max_box_w, text_w + 2 * pad_x)
    box_h = text_total_h + 2 * pad_y + 2 * safe_pad_y

    # Create transparent canvas.
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded (or flat) background — skip entirely when hidden (not alpha=0 box).
    if style.background_visible:
        bg = _hex_to_rgba(style.background_color, background_opacity)
        if _ROUNDED_RECT_OK:
            draw.rounded_rectangle([(0, 0), (box_w - 1, box_h - 1)], radius=radius, fill=bg)
        else:
            draw.rectangle([(0, 0), (box_w - 1, box_h - 1)], fill=bg)

    # Draw each text line centered horizontally. Start below the top safe pad
    # so diacritics never touch the rounded background edge.
    fg = _hex_to_rgb(style.text_color)
    y_cur = pad_y + safe_pad_y
    for ln in lines:
        lw, _ = _measure_text(ln, font)
        x_text = (box_w - lw) // 2
        draw.text((x_text, y_cur), ln, font=font, fill=fg)
        y_cur += line_h + line_gap

    img_path = os.path.join(tmp_dir, f"sub_{index:04d}.png")
    img.save(img_path, "PNG")

    # Overlay position: custom center (layout editor) or legacy bottom-center.
    if style.uses_custom_position():
        x_ratio = min(1.0, max(0.0, float(style.x_ratio)))
        y_ratio = min(1.0, max(0.0, float(style.y_ratio)))
        ov_x = round(video_w * x_ratio - box_w / 2)
        ov_y = round(video_h * y_ratio - box_h / 2)
    else:
        ov_x = (video_w - box_w) // 2
        ov_y = max(0, video_h - box_h - bottom_margin)
    ov_x, ov_y = _clamp_overlay_position(ov_x, ov_y, box_w, box_h, video_w, video_h)

    return _CueOverlay(image_path=img_path, x=ov_x, y=ov_y, start=0.0, end=0.0)


# ---------------------------------------------------------------------------
# Overlay position safety
# ---------------------------------------------------------------------------

def _clamp_overlay_position(
    ov_x: int,
    ov_y: int,
    box_w: int,
    box_h: int,
    display_w: int,
    display_h: int,
    safe_margin_x_ratio: float = 0.05,
) -> Tuple[int, int]:
    """Keep subtitle box inside the display frame with horizontal safe margins."""
    safe_margin_x = max(0, round(display_w * safe_margin_x_ratio))

    ov_x = max(safe_margin_x, ov_x)
    max_x = display_w - box_w - safe_margin_x
    if max_x >= safe_margin_x:
        ov_x = min(ov_x, max_x)
    else:
        ov_x = max(0, (display_w - box_w) // 2)

    ov_y = max(0, ov_y)
    ov_y = min(ov_y, max(0, display_h - box_h))
    return ov_x, ov_y


# ---------------------------------------------------------------------------
# Video size probe
# ---------------------------------------------------------------------------

def _get_video_size(video_path: str) -> Tuple[int, int, int, int, int]:
    """
    Return display dimensions and probe metadata for layout.

    Tuple: (display_width, display_height, stored_width, stored_height, rotation)
    """
    from .utils import get_video_display_size

    return get_video_display_size(video_path)


# ---------------------------------------------------------------------------
# Classic (ASS) fallback
# ---------------------------------------------------------------------------

def _burn_classic(
    video_path: str,
    srt_path: str,
    output_path: str,
    style: SubtitleRenderStyle,
) -> str:
    """
    Burn subtitles using the existing ASS / libass pipeline.

    Used as both the explicit "classic" mode and the automatic fallback when
    Pillow is unavailable or the rounded pipeline raises an exception.
    """
    import ffmpeg as _ffmpeg
    from .utils import prepare_burn_subtitles

    margin_pct = style.bottom_margin_ratio * 100.0
    # box_padding: approximate from padding_y scaled to reference_height
    box_padding = style.padding_y

    ass_path = prepare_burn_subtitles(
        srt_path,
        video_path,
        margin_pct,
        style.font_size,
        style.text_color,
        style.background_color,
        box_padding,
        style.reference_height,
        background_visible=style.background_visible,
        font_family=style.font_family,
        font_bold=style.font_bold,
    )

    try:
        vid = _ffmpeg.input(video_path)
        _ffmpeg.concat(
            vid.filter("subtitles", ass_path),
            vid.audio,
            v=1,
            a=1,
        ).output(output_path).run(quiet=True, overwrite_output=True)
    finally:
        try:
            os.remove(ass_path)
        except OSError:
            pass

    return output_path


# ---------------------------------------------------------------------------
# Rounded burn (Pillow + FFmpeg overlay)
# ---------------------------------------------------------------------------

def _burn_rounded(
    video_path: str,
    srt_path: str,
    output_path: str,
    style: SubtitleRenderStyle,
) -> str:
    """
    Internal implementation of the rounded-corner subtitle burn.

    Renders per-cue PNGs with Pillow, then chains them as FFmpeg overlays.
    Rotated videos are normalized to display orientation before overlay.
    Output rotation metadata is stripped so players do not double-rotate.
    """
    from .utils import (
        ffmpeg_remux_strip_rotation,
        ffmpeg_strip_rotation_metadata_args,
        ffmpeg_video_encode_args,
        log_video_orientation_probe,
        parse_srt,
        probe_video_orientation,
        validate_output_orientation,
        video_stream_has_rotation_metadata,
    )

    input_probe = log_video_orientation_probe("input_probe", video_path)

    video_w, video_h, stored_w, stored_h, rotation = _get_video_size(video_path)
    base_w, base_h = video_w, video_h
    # FFmpeg autorotate (default, no -noautorotate) presents [0:v] in display
    # orientation — equivalent to baking rotation before overlay, without
    # preserving stale Display Matrix metadata on the output stream.
    orientation_mode = (
        "autorotate (implicit)" if rotation else "none"
    )

    scale = video_h / max(style.reference_height, 1)
    scaled_font = max(12, round(style.font_size * scale))
    scaled_radius = max(4, round(style.border_radius * scale))
    scaled_pad = max(6, round(style.padding_x * scale))
    bottom_px = round(video_h * style.bottom_margin_ratio)

    probe_font, probe_path = _load_font(scaled_font, style.font_family, style.font_bold)
    try:
        ascent, descent = probe_font.getmetrics()
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        print(f"WARNING: Failed to read probe font metrics: {exc}")
        ascent, descent = 0, 0

    print(
        f"[Renderer] mode=rounded | "
        f"stored={stored_w}x{stored_h} | rotation={rotation}° | "
        f"display={video_w}x{video_h} | "
        f"ffmpeg_base_frame={base_w}x{base_h} | "
        f"orientation_normalize={orientation_mode} | "
        f"font_family={style.font_family} | "
        f"font_bold={style.font_bold} | "
        f"font_path={probe_path} | "
        f"font_size={scaled_font}px | "
        f"font_metrics=({ascent},{descent}) | "
        f"border_radius={scaled_radius}px | "
        f"padding={scaled_pad}px | "
        f"bottom_margin={bottom_px}px"
    )

    with open(srt_path, encoding="utf-8") as f:
        entries = parse_srt(f.read())

    tmp_dir = tempfile.mkdtemp(prefix="drakonsub_rnd_")

    try:
        overlays: List[_CueOverlay] = []
        for i, entry in enumerate(entries):
            text = entry.get("text", "").strip()
            if not text:
                continue
            ov = _render_cue_image(text, style, video_w, video_h, tmp_dir, i)
            if ov is None:
                continue
            ov.start = _parse_ts(entry["start_str"])
            ov.end = _parse_ts(entry["end_str"])
            overlays.append(ov)

        if not overlays:
            return _burn_classic(video_path, srt_path, output_path, style)

        if overlays:
            first = overlays[0]
            from PIL import Image as _PILImage
            with _PILImage.open(first.image_path) as _img:
                _ow, _oh = _img.size
            print(
                f"[Renderer] first_overlay x={first.x} y={first.y} "
                f"box={_ow}x{_oh} (on {base_w}x{base_h} frame)"
            )

        # Autorotate on decode (default): [0:v] matches display dimensions.
        cmd = ["ffmpeg", "-y", "-i", video_path]
        for ov in overlays:
            cmd += ["-i", ov.image_path]

        parts: List[str] = []
        prev = "[0:v]"

        for j, ov in enumerate(overlays):
            inp = f"[{j + 1}:v]"
            out = "[vout]" if j == len(overlays) - 1 else f"[v{j}]"
            parts.append(
                f"{prev}{inp}overlay={ov.x}:{ov.y}"
                f":enable='between(t,{ov.start:.3f},{ov.end:.3f})'{out}"
            )
            prev = out

        filter_str = ";".join(parts)
        preview = filter_str if len(filter_str) <= 500 else filter_str[:500] + "..."
        print(f"[Renderer] filtergraph={preview}")

        filter_script: Optional[str] = None
        if len(filter_str) > 80_000:
            filter_script = os.path.join(tmp_dir, "filter.txt")
            with open(filter_script, "w", encoding="utf-8") as fp:
                fp.write(filter_str)
            cmd += ["-filter_complex_script", filter_script]
        else:
            cmd += ["-filter_complex", filter_str]

        cmd += ["-map", "[vout]", "-map", "0:a?"]
        cmd += ffmpeg_video_encode_args()
        cmd += ["-c:a", "copy"]
        cmd += ffmpeg_strip_rotation_metadata_args()
        cmd.append(output_path)

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-600:] if proc.stderr else "FFmpeg error")

        log_video_orientation_probe("output_probe", output_path)
        metadata_stripped = not video_stream_has_rotation_metadata(output_path)
        print(f"[Renderer] metadata_strip_succeeded={metadata_stripped}")

        if not metadata_stripped:
            remux_path = os.path.join(tmp_dir, "remux.mp4")
            print("[Renderer] fallback_remux=true (stale rotation metadata detected)")
            remux_cmd = ffmpeg_remux_strip_rotation(output_path, remux_path)
            remux_proc = subprocess.run(remux_cmd, capture_output=True, text=True)
            if remux_proc.returncode != 0:
                raise RuntimeError(
                    remux_proc.stderr[-600:] if remux_proc.stderr else "Remux failed"
                )
            shutil.move(remux_path, output_path)
            log_video_orientation_probe("output_probe_after_remux", output_path)
            metadata_stripped = not video_stream_has_rotation_metadata(output_path)
            print(f"[Renderer] metadata_strip_succeeded={metadata_stripped}")
        else:
            print("[Renderer] fallback_remux=false")

        if not metadata_stripped:
            raise RuntimeError(
                "Output still contains rotation metadata after render and remux"
            )

        validate_output_orientation(video_path, output_path)
        out_probe = probe_video_orientation(output_path)
        print(
            f"[Renderer] orientation_validation=passed | "
            f"input_display={input_probe['display_width']}x"
            f"{input_probe['display_height']} "
            f"{input_probe['visual_orientation']} -> "
            f"output_display={out_probe['display_width']}x"
            f"{out_probe['display_height']} "
            f"{out_probe['visual_orientation']}"
        )

        return output_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def burn_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    style: Optional[SubtitleRenderStyle] = None,
) -> str:
    """
    Burn subtitles into *video_path*, writing the result to *output_path*.

    Chooses the render mode from *style.mode*:
      "rounded" — Pillow rounded-rectangle overlay (modern look).
      "classic"  — existing ASS / libass pipeline (rectangular box).

    Always falls back to classic mode when Pillow is unavailable or any
    error occurs in the rounded pipeline, so the application never crashes.

    Parameters
    ----------
    video_path:   Path to the input video file.
    srt_path:     Path to the (Vietnamese) SRT file with final text + timing.
    output_path:  Destination path for the output video.
    style:        Render style config; loaded from env when ``None``.

    Returns
    -------
    str   Path of the written output video (always *output_path* on success).
    """
    if style is None:
        style = load_render_style()

    if style.mode != "rounded":
        print(f"[Renderer] mode=classic")
        return _burn_classic(video_path, srt_path, output_path, style)

    if not _PILLOW_OK:
        print("[Renderer] Pillow not installed — falling back to classic mode.")
        return _burn_classic(video_path, srt_path, output_path, style)

    try:
        return _burn_rounded(video_path, srt_path, output_path, style)
    except Exception as exc:
        print(f"[Renderer] Rounded mode failed ({exc}) — falling back to classic.")
        return _burn_classic(video_path, srt_path, output_path, style)

"""
visualize_run.py — Turn a previous Video Explorer run into a replay video.

Frames are shown in TRUE RUN ORDER using the leading sequence number in
filenames, so the video replays exactly what the system did, step by step.

The right-hand side shows a LIVE SCRATCHPAD PANEL that updates whenever
a new scratchpad_* image is detected — giving a real-time view of the
evidence collected so far.  The question being answered is pinned at the
top of this panel from frame one, so viewers always have context for what
the system is looking for.

A full-width TIMELINE SCRUBBER at the bottom shows exactly where in the
exploration process each frame falls, color-coded by activity type, so
viewers always know where they are in the overall process.

Visual identity follows DESIGN.md: dark navy chrome, blue→cyan gradient
accents, brand mark in the header, monochrome-leaning category palette.

Usage:
    python visualize_run.py
    python visualize_run.py --run results/run_20240101_120000_images
    python visualize_run.py --run results/run_20240101_120000_images --fps 1.5
    python visualize_run.py --run results/run_20240101_120000_images --result results/result_20240101_120000.json

Output:
    <run_folder>/replay.mp4
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time as _time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ==========================================
# LAYOUT CONSTANTS
# ==========================================
HEADER_HEIGHT    = 86    # Top bar: brand + activity + progress strip
TIMELINE_HEIGHT  = 34    # Bottom bar: phase scrubber
MAIN_WIDTH       = 960   # Left panel (main frame)
SCRATCHPAD_WIDTH = 580   # Right panel (question + evidence log)
OUTPUT_WIDTH     = MAIN_WIDTH + SCRATCHPAD_WIDTH   # 1540


# ==========================================
# BRAND TOKENS  (mirrors DESIGN.md — dark mode)
# ==========================================
BRAND_BLUE       = (59, 130, 246)    # #3B82F6
BRAND_CYAN       = (6,  182, 212)    # #06B6D4

BG_PRIMARY       = (15,  23,  42)    # #0F172A
BG_SECONDARY     = (30,  41,  59)    # #1E293B
BG_TERTIARY      = (51,  65,  85)    # #334155

TEXT_PRIMARY     = (249, 250, 251)   # #F9FAFB
TEXT_SECONDARY   = (203, 213, 225)   # #CBD5E1
TEXT_TERTIARY    = (148, 163, 184)   # #94A3B8

BORDER_SUBTLE    = (40,  52,  73)    # near-invisible on dark navy
BORDER_DEFAULT   = (51,  65,  85)    # divider color

SUCCESS_GREEN    = (16,  185, 129)   # #10B981


# ==========================================
# FRAME CLASSIFICATION  (label + colour only)
# ==========================================
CATEGORIES = [
    (r"^global_grid",                    "global",      "Global Grid"),
    (r"^DFS_round(\d+)_masked_grid",     "dfs_masked",  "Scanning Regions"),
    (r"^DFS_round(\d+)_uncertainty",     "dfs_uncert",  "Analyzing Confidence"),
    (r"^W(\d+)_C(\d+)_step(\d+)",        "worker",      "Inspecting Details"),
    (r"^BFS_batch(\d+)_masked_grid",     "bfs_masked",  "Tracking Explored Areas"),
    (r"^BFS_batch(\d+)_uncertainty",     "bfs_uncert",  "Re-evaluating Targets"),
    (r"^BFSW(\d+)_depth(\d+)_step(\d+)","bfsworker",   "Deep Dive Investigation"),
    (r"^zoom_",                          "zoom",        "Zooming In"),
    (r"^grid_c(.+?)s_span(.+?)s",        "navgrid",     "Navigating Timeline"),
    (r"^scratchpad_(\d+)items",          "scratchpad",  "Reviewing Evidence"),
]

CATEGORY_DESCRIPTIONS = {
    "global":     "Getting a bird's-eye view of the entire video",
    "dfs_masked": "Deciding which parts of the video need closer inspection",
    "dfs_uncert": "Mapping out the most critical moments",
    "worker":     "Zooming in on a specific moment to examine it closely",
    "bfs_masked": "Marking which areas of the video have already been explored",
    "bfs_uncert": "Updating the search based on new findings",
    "bfsworker":  "Drilling deeper into a promising region of the video",
    "zoom":       "Examining a specific timestamp up close",
    "navgrid":    "Navigating the video timeline to choose the next region",
    "scratchpad": "Reviewing all the evidence collected so far",
    "unknown":    "Processing…",
}

# Brand-aligned palette: a cool-family continuum (blue → cyan → teal) for
# exploration activities, with success green reserved for evidence and slate
# for neutral/transit states.  All hues live inside the DESIGN.md spirit of
# "quiet confidence — no decoration for its own sake".
CATEGORY_COLORS = {
    "global":     BRAND_BLUE,             # anchor / overview
    "dfs_masked": (79,  114, 220),        # blue, slightly muted — planning
    "dfs_uncert": (96,  165, 250),        # lighter blue — analysis
    "worker":     BRAND_CYAN,             # active exploration
    "bfs_masked": (20,  184, 166),        # teal — alternate mode
    "bfs_uncert": (45,  212, 191),        # teal light
    "bfsworker":  (14,  165, 233),        # sky — deep dive
    "zoom":       (34,  211, 238),        # bright cyan — focus
    "navgrid":    (100, 116, 139),        # slate — transit
    "scratchpad": SUCCESS_GREEN,          # evidence accumulated
    "unknown":    (71,  85,  105),
}


def _rgb_to_bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


def classify(filename):
    """Return (category, label) for a filename."""
    stem = Path(filename).stem
    base = re.sub(r'_\d{3}$', '', stem)

    for pattern, category, label_tpl in CATEGORIES:
        m = re.match(pattern, base, re.IGNORECASE)
        if m:
            label = label_tpl
            for i, g in enumerate(m.groups(), 1):
                label = label.replace(f"{{{i}}}", str(g))
            return category, label

    return "unknown", base


# ==========================================
# FONT, LOGO & DRAW HELPERS
# ==========================================
_FONT_CACHE = {}

def _get_font(size, bold=False):
    """Brand prefers Poppins/Inter Bold for headings.  Fall back gracefully."""
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    bold_candidates = [
        "Poppins-Bold.ttf", "Poppins-SemiBold.ttf",
        "Inter-Bold.ttf",   "Inter-SemiBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    regular_candidates = [
        "Poppins-Regular.ttf",
        "Inter-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    candidates = bold_candidates if bold else regular_candidates

    for font_path in candidates:
        try:
            font = ImageFont.truetype(font_path, size)
            _FONT_CACHE[key] = font
            return font
        except Exception:
            continue

    font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


_LOGO_CACHE = {}

def _load_logo(target_height):
    """Load the brand mark, auto-crop transparent padding, cache by target height."""
    if target_height in _LOGO_CACHE:
        return _LOGO_CACHE[target_height]

    logo_path = Path(__file__).parent / "figures" / "logo-videoatlas.png"
    if not logo_path.exists():
        _LOGO_CACHE[target_height] = None
        return None

    try:
        img  = Image.open(logo_path).convert("RGBA")
        bbox = img.getbbox()              # crops out fully-transparent borders
        if bbox:
            img = img.crop(bbox)
        # The source asset is a horizontal lockup: brand mark on the left,
        # faint wordmark on the right.  We render the wordmark separately with
        # clean typography on dark navy, so isolate the mark by scanning for
        # the empty vertical gap between mark and wordmark using alpha density.
        arr = np.array(img)
        if arr.shape[2] == 4:
            alpha       = arr[:, :, 3]
            col_density = (alpha > 30).sum(axis=0) / max(1, alpha.shape[0])
            min_x       = int(alpha.shape[1] * 0.10)
            gap_x       = None
            in_gap_len  = 0
            for x in range(min_x, alpha.shape[1]):
                if col_density[x] < 0.02:
                    in_gap_len += 1
                    if in_gap_len >= 6:    # require a sustained gap
                        gap_x = x - in_gap_len + 1
                        break
                else:
                    in_gap_len = 0
            if gap_x is not None and gap_x > min_x:
                img = img.crop((0, 0, gap_x, alpha.shape[0]))
                # Re-tighten vertically in case the spotlight glow is asymmetric
                inner = img.getbbox()
                if inner:
                    img = img.crop(inner)
        w, h  = img.size
        new_w = max(1, int(w * target_height / h))
        img   = img.resize((new_w, target_height), Image.LANCZOS)
        _LOGO_CACHE[target_height] = img
        return img
    except Exception:
        _LOGO_CACHE[target_height] = None
        return None


def _draw_gradient_rect(draw, x1, y1, x2, y2, color_a, color_b):
    """Horizontal gradient from color_a (left) → color_b (right)."""
    if x2 <= x1:
        return
    span = x2 - x1
    for i in range(span):
        t = i / max(1, span - 1)
        r = int(color_a[0] + (color_b[0] - color_a[0]) * t)
        g = int(color_a[1] + (color_b[1] - color_a[1]) * t)
        b = int(color_a[2] + (color_b[2] - color_a[2]) * t)
        draw.line([(x1 + i, y1), (x1 + i, y2)], fill=(r, g, b))


def _format_time(seconds):
    """Convert raw seconds into MM:SS for laymen."""
    try:
        seconds = float(seconds)
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
    except (ValueError, TypeError):
        return "00:00"


def _wrap_text(text, font, max_width, draw):
    """Split *text* into lines that fit within *max_width* pixels."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = (current + " " + word).strip()
        w     = draw.textlength(trial, font=font)
        if w <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


# ==========================================
# REASONING SIDECAR LOADER
# ==========================================
def _load_reasoning_for_scratchpad(scratchpad_img_path):
    """
    Given a path like  .../0042_scratchpad_5items.jpg
    look for           .../0042_scratchpad_5items_reasoning.json
    Returns list of dicts or [] if not found.
    """
    if scratchpad_img_path is None:
        return []
    p         = Path(scratchpad_img_path)
    json_path = p.parent / (p.stem + "_reasoning.json")
    if not json_path.exists():
        return []
    try:
        with open(json_path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


# ==========================================
# SCRATCHPAD PANEL
# ==========================================
def build_scratchpad_panel(scratchpad_img_path, prev_sp_path,
                           panel_w, panel_h, result=None, question=None):
    """
    Build a fixed-size scratchpad panel (numpy BGR array).

    Layout (top → bottom):
      • QUESTION card         (pinned from frame 1)
      • FINAL ANSWER card     (revealed after the AI concludes)
      • EVIDENCE LOG header   (with count pill)
      • Reasoning cards       (newest first; older items collapsed under a
                               "↑ N hidden" indicator if they overflow)
    """
    panel = np.full((panel_h, panel_w, 3), BG_SECONDARY, dtype=np.uint8)
    pil   = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw  = ImageDraw.Draw(pil)

    PAD          = 16
    CARD_PAD     = 16
    CARD_RADIUS  = 12

    font_label    = _get_font(10, bold=True)
    font_body     = _get_font(16)
    font_section  = _get_font(12, bold=True)
    font_ans_main = _get_font(22, bold=True)
    font_ev_hdr   = _get_font(13, bold=True)
    font_ev_meta  = _get_font(11)
    font_ev_body  = _get_font(13)

    y = PAD

    # ── QUESTION card ─────────────────────────────────────────────
    if question:
        q_lines = _wrap_text(question, font_body,
                              panel_w - PAD * 2 - CARD_PAD * 2, draw)
        card_h  = CARD_PAD + 12 + 10 + len(q_lines) * 22 + CARD_PAD
        x1, y1  = PAD, y
        x2, y2  = panel_w - PAD, y + card_h

        draw.rounded_rectangle([x1, y1, x2, y2],
                                radius=CARD_RADIUS, fill=BG_TERTIARY)
        # Cyan accent stripe on top edge
        draw.rounded_rectangle([x1, y1, x2, y1 + 3], radius=2, fill=BRAND_CYAN)

        draw.text((x1 + CARD_PAD, y1 + CARD_PAD), "QUESTION",
                   font=font_label, fill=TEXT_TERTIARY)
        ty = y1 + CARD_PAD + 12 + 10
        for ql in q_lines:
            draw.text((x1 + CARD_PAD, ty), ql, font=font_body, fill=TEXT_PRIMARY)
            ty += 22
        y = y2 + 12

    # ── FINAL ANSWER card ─────────────────────────────────────────
    if result is not None:
        answer    = result.get("predicted_answer", "?")
        ans_lines = _wrap_text(str(answer), font_ans_main,
                                panel_w - PAD * 2 - CARD_PAD * 2, draw)
        card_h    = CARD_PAD + 12 + 12 + len(ans_lines) * 28 + CARD_PAD
        x1, y1    = PAD, y
        x2, y2    = panel_w - PAD, y + card_h

        draw.rounded_rectangle([x1, y1, x2, y2],
                                radius=CARD_RADIUS, fill=BG_TERTIARY)
        # Brand blue→cyan gradient across the top — the moment of conclusion
        _draw_gradient_rect(draw, x1, y1, x2, y1 + 4, BRAND_BLUE, BRAND_CYAN)

        draw.text((x1 + CARD_PAD, y1 + CARD_PAD), "FINAL ANSWER",
                   font=font_label, fill=TEXT_TERTIARY)
        ty = y1 + CARD_PAD + 12 + 12
        for line in ans_lines:
            draw.text((x1 + CARD_PAD, ty), line,
                       font=font_ans_main, fill=TEXT_PRIMARY)
            ty += 28
        y = y2 + 16

    # ── EVIDENCE LOG section header ───────────────────────────────
    reasoning_items = _load_reasoning_for_scratchpad(scratchpad_img_path)
    prev_items      = _load_reasoning_for_scratchpad(prev_sp_path)
    new_count       = max(0, len(reasoning_items) - len(prev_items))

    if scratchpad_img_path is None:
        pill_text = "collecting"
    else:
        m         = re.search(r'scratchpad_(\d+)items',
                              Path(scratchpad_img_path).stem)
        n_items   = m.group(1) if m else "?"
        pill_text = f"{n_items} item{'s' if n_items != '1' else ''}"

    draw.text((PAD + 2, y), "EVIDENCE LOG",
               font=font_section, fill=TEXT_SECONDARY)
    pill_w  = int(draw.textlength(pill_text, font=font_label)) + 16
    pill_x2 = panel_w - PAD
    pill_x1 = pill_x2 - pill_w
    draw.rounded_rectangle([pill_x1, y - 1, pill_x2, y + 17],
                            radius=9, fill=BG_TERTIARY)
    pill_tx = pill_x1 + (pill_w - int(draw.textlength(pill_text, font=font_label))) // 2
    draw.text((pill_tx, y + 3), pill_text,
               font=font_label, fill=TEXT_SECONDARY)
    y += 30

    if not reasoning_items:
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # ── Reasoning cards ───────────────────────────────────────────
    blocks = []
    for idx, item in enumerate(reasoning_items):
        is_new      = new_count > 0 and idx >= len(reasoning_items) - new_count
        letter      = item.get("letter", "?")
        t           = item.get("time", 0.0)
        desc        = item.get("description", "")
        header_text = f"Event {letter}"
        meta_text   = _format_time(t)
        desc_lines  = _wrap_text(desc, font_ev_body,
                                  panel_w - PAD * 2 - CARD_PAD * 2 - 6, draw)
        block_h = CARD_PAD + 16 + 8 + len(desc_lines) * 18 + CARD_PAD - 4
        blocks.append({
            "is_new": is_new, "header": header_text, "meta": meta_text,
            "lines": desc_lines, "h": block_h,
        })

    avail_h = panel_h - y - PAD
    total_h = sum(b["h"] + 10 for b in blocks)
    start   = 0
    if total_h > avail_h:
        running = 0
        for i, b in enumerate(blocks):
            running += b["h"] + 10
            if running >= total_h - avail_h:
                start = i + 1
                break

    if start > 0:
        font_trunc = _get_font(10)
        msg        = f"{start} earlier item{'s' if start != 1 else ''} hidden"
        draw.text((PAD + 2, y), "↑  " + msg,
                   font=font_trunc, fill=TEXT_TERTIARY)
        y += 18

    for b in blocks[start:]:
        if y + b["h"] > panel_h - PAD:
            break
        x1, y1 = PAD, y
        x2, y2 = panel_w - PAD, y + b["h"]
        draw.rounded_rectangle([x1, y1, x2, y2], radius=10, fill=BG_TERTIARY)
        # Cyan left-edge accent on items added in this round
        if b["is_new"]:
            draw.rounded_rectangle([x1, y1, x1 + 3, y2],
                                    radius=2, fill=BRAND_CYAN)

        draw.text((x1 + CARD_PAD, y1 + CARD_PAD), b["header"],
                   font=font_ev_hdr, fill=TEXT_PRIMARY)
        meta_w = int(draw.textlength(b["meta"], font=font_ev_meta))
        draw.text((x2 - CARD_PAD - meta_w, y1 + CARD_PAD + 2),
                   b["meta"], font=font_ev_meta, fill=TEXT_TERTIARY)

        ty = y1 + CARD_PAD + 16 + 8
        for line in b["lines"]:
            draw.text((x1 + CARD_PAD, ty), line,
                       font=font_ev_body, fill=TEXT_SECONDARY)
            ty += 18

        y = y2 + 10

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ==========================================
# HEADER
# ==========================================
def make_header(label, category, frame_num, total_frames):
    color_rgb = CATEGORY_COLORS.get(category, CATEGORY_COLORS["unknown"])
    desc      = CATEGORY_DESCRIPTIONS.get(category, label)

    header = np.full((HEADER_HEIGHT, OUTPUT_WIDTH, 3), BG_PRIMARY, dtype=np.uint8)
    pil    = Image.fromarray(cv2.cvtColor(header, cv2.COLOR_BGR2RGB))
    draw   = ImageDraw.Draw(pil)

    inner_top    = 0
    inner_bottom = HEADER_HEIGHT - 4    # leave room for progress strip
    cx_y         = (inner_top + inner_bottom) // 2

    # ── Brand block ───────────────────────────────────────────────
    logo_h   = 44
    logo_y   = cx_y - logo_h // 2
    logo_img = _load_logo(logo_h)
    cursor_x = 20

    if logo_img is not None:
        pil.paste(logo_img, (cursor_x, logo_y), logo_img)
        cursor_x += logo_img.size[0] + 12

    font_brand = _get_font(20, bold=True)
    brand      = "VideoAtlas"
    brand_w    = int(draw.textlength(brand, font=font_brand))
    # Vertically center the wordmark inside the chrome
    draw.text((cursor_x, cx_y - 13), brand,
               font=font_brand, fill=TEXT_PRIMARY)
    cursor_x += brand_w + 22

    # Vertical divider
    div_top    = cx_y - 14
    div_bottom = cx_y + 14
    draw.line([(cursor_x, div_top), (cursor_x, div_bottom)],
               fill=BORDER_DEFAULT, width=1)
    cursor_x += 18

    # ── Category dot ──────────────────────────────────────────────
    dot_r = 5
    draw.ellipse([cursor_x, cx_y - dot_r,
                   cursor_x + dot_r * 2, cx_y + dot_r],
                  fill=color_rgb)
    cursor_x += dot_r * 2 + 10

    # ── Activity description + frame counter ──────────────────────
    font_desc  = _get_font(15, bold=True)
    font_meta  = _get_font(12)

    frame_text = f"Frame {frame_num} / {total_frames}"
    meta_w     = int(draw.textlength(frame_text, font=font_meta))
    max_desc_w = OUTPUT_WIDTH - cursor_x - meta_w - 40

    desc_render = desc
    while int(draw.textlength(desc_render, font=font_desc)) > max_desc_w and len(desc_render) > 4:
        desc_render = desc_render[:-2]
    if desc_render != desc:
        desc_render = desc_render.rstrip() + "…"

    draw.text((cursor_x, cx_y - 10), desc_render,
               font=font_desc, fill=TEXT_PRIMARY)
    draw.text((OUTPUT_WIDTH - 20 - meta_w, cx_y - 8),
               frame_text, font=font_meta, fill=TEXT_TERTIARY)

    # ── Progress bar — full-width gradient on a subtle track ──────
    bar_y, bar_h = HEADER_HEIGHT - 4, 4
    draw.rectangle([(0, bar_y), (OUTPUT_WIDTH, bar_y + bar_h)],
                    fill=BORDER_SUBTLE)
    fill_x = int(OUTPUT_WIDTH * frame_num / max(1, total_frames))
    if fill_x > 0:
        _draw_gradient_rect(draw, 0, bar_y, fill_x, bar_y + bar_h,
                             BRAND_BLUE, BRAND_CYAN)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ==========================================
# TIMELINE SCRUBBER
# ==========================================
def build_timeline_scrubber(categories_all, current_idx, width, height):
    """
    Full-width bar at the bottom of the frame.  Each segment represents one
    frame, colored by its activity category.  Past frames are dimmed; the
    current frame is marked with a bright vertical tick.  Wrapped in a
    subtle rounded track so it reads as a single composed element rather
    than a row of raw rectangles.
    """
    bar = np.full((height, width, 3), BG_PRIMARY, dtype=np.uint8)
    if not categories_all:
        return bar

    pil  = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    # Label
    label_font = _get_font(10, bold=True)
    draw.text((16, (height - 12) // 2), "TIMELINE",
               font=label_font, fill=TEXT_TERTIARY)

    # Track
    inner_x1 = 92
    inner_x2 = width - 16
    inner_y1 = 9
    inner_y2 = height - 9
    inner_w  = inner_x2 - inner_x1
    inner_h  = inner_y2 - inner_y1
    draw.rounded_rectangle([inner_x1, inner_y1, inner_x2, inner_y2],
                            radius=inner_h // 2, fill=BG_SECONDARY)

    total = len(categories_all)
    seg_w = inner_w / total
    for i, cat in enumerate(categories_all):
        color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["unknown"])
        x1    = inner_x1 + int(i * seg_w)
        x2    = inner_x1 + max(int(i * seg_w) + 1, int((i + 1) * seg_w))
        if i < current_idx:
            color = (int(color[0] * 0.45),
                     int(color[1] * 0.45),
                     int(color[2] * 0.45))
        draw.rectangle([x1, inner_y1 + 2, x2, inner_y2 - 2], fill=color)

    # Current-position marker
    cx = inner_x1 + int((current_idx + 0.5) * seg_w)
    draw.rectangle([cx - 1, inner_y1 - 2, cx + 1, inner_y2 + 2],
                    fill=TEXT_PRIMARY)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ==========================================
# FRAME ASSEMBLY
# ==========================================
def prepare_frame(img_path, label, category, frame_num, total_frames,
                  target_content_h, scratchpad_img_path, prev_sp_path,
                  result=None, question=None,
                  categories_all=None, current_idx=0):
    """
    Build one output frame:
        header          (HEADER_HEIGHT)
        main | scratchpad  (target_content_h)
        timeline scrubber  (TIMELINE_HEIGHT)
    """
    pad_color_bgr = _rgb_to_bgr(BG_PRIMARY)

    # ---- Main frame ----
    img = cv2.imread(str(img_path))
    if img is None:
        img = np.full((target_content_h, MAIN_WIDTH, 3), pad_color_bgr, dtype=np.uint8)
        cv2.putText(img, f"Could not load: {img_path.name}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _rgb_to_bgr((239, 68, 68)), 2)
    else:
        # Scale-to-fit inside MAIN_WIDTH × target_content_h preserving aspect.
        # Never crop — tall grids (8×8) must stay fully visible.  Excess space
        # is letterboxed with the chrome background colour so it reads as part
        # of the surrounding panel rather than a hard black bar.
        h, w  = img.shape[:2]
        scale = min(MAIN_WIDTH / w, target_content_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img   = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((target_content_h, MAIN_WIDTH, 3),
                          pad_color_bgr, dtype=np.uint8)
        y_off  = (target_content_h - new_h) // 2
        x_off  = (MAIN_WIDTH - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = img
        img = canvas

    # ---- Scratchpad panel ----
    sp_panel = build_scratchpad_panel(
        scratchpad_img_path, prev_sp_path,
        SCRATCHPAD_WIDTH, target_content_h,
        result=result, question=question,
    )

    # ---- Vertical divider — subtle, matches border token ----
    divider = np.full((target_content_h, 1, 3),
                       _rgb_to_bgr(BORDER_DEFAULT), dtype=np.uint8)

    content_row = np.hstack([img, divider, sp_panel])
    if content_row.shape[1] != OUTPUT_WIDTH:
        content_row = cv2.resize(content_row, (OUTPUT_WIDTH, target_content_h))

    header   = make_header(label, category, frame_num, total_frames)
    scrubber = build_timeline_scrubber(
        categories_all or [], current_idx, OUTPUT_WIDTH, TIMELINE_HEIGHT,
    )

    return np.vstack([header, content_row, scrubber])


# ==========================================
# SCRATCHPAD TIMELINE BUILDER
# ==========================================
def build_scratchpad_timeline(classified):
    """
    For each frame index i return (current_sp_path, prev_sp_path).
    current_sp_path — most recent scratchpad seen at or before frame i.
    prev_sp_path    — the scratchpad before current_sp_path, used to diff
                      and highlight newly added evidence items.
    """
    timeline = []
    prev     = None
    current  = None
    for category, _label, img_path in classified:
        if category == "scratchpad":
            prev    = current
            current = img_path
        timeline.append((current, prev))
    return timeline


# ==========================================
# RESULT JSON LOADER
# ==========================================
def load_result(run_folder):
    run_folder  = Path(run_folder)
    parent      = run_folder.parent
    stem        = run_folder.stem
    timestamp   = stem.replace("run_", "").replace("_images", "")
    result_path = parent / f"result_{timestamp}.json"

    if result_path.exists():
        with open(result_path) as f:
            return json.load(f)

    candidates = sorted(parent.glob("result_*.json"))
    if candidates:
        with open(candidates[-1]) as f:
            return json.load(f)

    return None


def format_result_info(result):
    if not result:
        return None
    choice = result.get("predicted_choice", -1)
    answer = result.get("predicted_answer", "?")
    q      = result.get("question", "")[:40]
    return f"Q: {q}…  →  [{choice}] {answer[:30]}"


# ==========================================
# H.264 RE-ENCODE
# ==========================================
def _reencode_h264(input_path, output_path, end_hold_seconds=4):
    # tpad=stop_mode=clone:stop_duration=N freezes the last frame for N seconds.
    # Using a list avoids shell injection from paths with spaces or quotes.
    vf  = f"tpad=stop_mode=clone:stop_duration={end_hold_seconds}"
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
        str(output_path),
    ]
    print(f"  Re-encoding to H.264 with ffmpeg (+ {end_hold_seconds}s end freeze)...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print("  [!] ffmpeg not found on PATH — skipping re-encode pass.")
        return False
    if result.returncode != 0:
        print(f"  [!] ffmpeg re-encode failed:\n{result.stderr[-400:]}")
        return False
    return True


# ==========================================
# BUILD VIDEO  (public API used by main.py)
# ==========================================
def build_video(run_folder, output_path=None, fps=1.5, result_json_path=None):
    run_folder = Path(run_folder)

    if not run_folder.is_dir():
        print(f"[!] Folder not found: {run_folder}")
        return None

    all_images = list(run_folder.glob("*.jpg"))
    if not all_images:
        print(f"[!] No .jpg images found in: {run_folder}")
        return None

    print(f"\nFound {len(all_images)} images in: {run_folder}")

    def _seq(p):
        m = re.match(r'^(\d+)_', p.name)
        return int(m.group(1)) if m else 0

    all_images.sort(key=_seq)
    print("  Sorted by sequence number (true run order).")

    classified = []
    for img_path in all_images:
        bare_name = re.sub(r'^\d+_', '', img_path.name)
        category, label = classify(bare_name)
        classified.append((category, label, img_path))

    sp_timeline    = build_scratchpad_timeline(classified)
    categories_all = [c for c, _, _ in classified]

    if result_json_path:
        with open(result_json_path) as f:
            result = json.load(f)
    else:
        result = load_result(run_folder)

    question = result.get("question", "") if result else ""

    if result:
        print(f"  Result: [{result.get('predicted_choice')}] "
              f"{result.get('predicted_answer', '')[:60]}")
    else:
        print("  No result JSON found — answer overlay disabled.")

    if question:
        print(f"  Question pinned from frame 1: {question[:80]}")

    # Find the last scratchpad frame — answer box revealed from that point on.
    last_sp_idx = None
    for idx, (category, _, _) in enumerate(classified):
        if category == "scratchpad":
            last_sp_idx = idx
    if last_sp_idx is not None:
        print(f"  Answer will appear from frame {last_sp_idx + 1} / {len(classified)} "
              f"(after the final scratchpad).")

    if output_path is None:
        output_path = str(run_folder / "replay.mp4")

    total = len(classified)
    print(f"\nBuilding video: {total} frames @ {fps} fps  →  {output_path}")
    print(f"  Layout: {MAIN_WIDTH}px (main) + {SCRATCHPAD_WIDTH}px (scratchpad) = {OUTPUT_WIDTH}px wide")

    sample_heights = []
    for _, _, img_path in classified[:min(20, total)]:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            sample_heights.append(int(h * MAIN_WIDTH / w))
    # Use the TALLEST sample so the panel can show every image at full size —
    # square 8×8 grids especially must never be cropped.  Shorter images
    # (scratchpads, zooms) get centered with letterbox padding in prepare_frame.
    target_content_h = max(sample_heights) if sample_heights else 720
    # H.264 (avc1) requires even dimensions; round up to keep the layout intact.
    if target_content_h % 2 != 0:
        target_content_h += 1
    total_h          = target_content_h + HEADER_HEIGHT + TIMELINE_HEIGHT
    if total_h % 2 != 0:
        total_h += 1

    # opencv-python wheels vary in which codecs they register (e.g. macOS builds
    # without FFmpeg drop mp4v).  Try a sequence of fourccs and use the first
    # one that opens.  avc1 = H.264 via AVFoundation/V4L, MJPG = universal.
    tmp_path = output_path.replace(".mp4", "_raw.mp4")
    writer = None
    for fcc in ("avc1", "mp4v", "H264", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*fcc)
        # MJPG needs a .avi container to be reliable
        candidate_path = tmp_path
        if fcc == "MJPG":
            candidate_path = tmp_path.replace(".mp4", ".avi")
        # Wipe any leftover from a previous (possibly failed) attempt so the
        # codec gets a clean slate.
        if os.path.exists(candidate_path):
            try:
                os.remove(candidate_path)
            except OSError:
                pass
        w = cv2.VideoWriter(candidate_path, fourcc, fps, (OUTPUT_WIDTH, total_h))
        if w.isOpened():
            writer   = w
            tmp_path = candidate_path
            print(f"  Encoder: {fcc}")
            break
        w.release()

    if writer is None:
        print("[!] Could not open VideoWriter with any of avc1/mp4v/H264/MJPG.")
        print("    Reinstall opencv-python (`pip install --force-reinstall opencv-python`).")
        return None

    for i, (category, label, img_path) in enumerate(classified, 1):
        sp_path, prev_sp     = sp_timeline[i - 1]
        result_for_frame     = result if (last_sp_idx is None or (i - 1) >= last_sp_idx) else None
        frame = prepare_frame(
            img_path, label, category, i, total,
            target_content_h, sp_path, prev_sp,
            result=result_for_frame, question=question,
            categories_all=categories_all, current_idx=i - 1,
        )
        writer.write(frame)

        if i % 10 == 0 or i == total:
            sp_status = f"  [SP: {sp_path.name[:30]}]" if sp_path else ""
            print(f"  [{i:4d}/{total}]  {label[:60]}{sp_status}")

    writer.release()

    ffmpeg_ok = _reencode_h264(tmp_path, output_path)
    if ffmpeg_ok:
        os.remove(tmp_path)
        print(f"\n✓  Saved (H.264 + end freeze): {output_path}")
    else:
        # Fallback: keep what cv2.VideoWriter produced.  If the encoder needed
        # an .avi container (MJPG), preserve the extension so the file stays
        # playable rather than mis-labeled as .mp4.
        import shutil
        src_ext  = os.path.splitext(tmp_path)[1]
        out_ext  = os.path.splitext(output_path)[1]
        final_out = output_path
        if src_ext != out_ext:
            final_out = os.path.splitext(output_path)[0] + src_ext
        shutil.move(tmp_path, final_out)
        print(f"\n✓  Saved (no ffmpeg re-encode): {final_out}")
        print("   Install ffmpeg to add the end-of-video freeze frame.")
        output_path = final_out

    duration = total / fps
    print(f"   {total} frames · {fps} fps · {duration:.1f}s duration")
    return output_path


# ==========================================
# CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate a replay video from a Video Explorer debug images folder."
    )
    parser.add_argument("--run",    "-r", default=None,
                        help="Path to the run images folder.")
    parser.add_argument("--output", "-o", default=None,
                        help="Output MP4 path (default: <run_folder>/replay.mp4).")
    parser.add_argument("--fps",    "-f", type=float, default=1.5,
                        help="Frames per second (default: 1.5). Use 0.5 for slow playback.")
    parser.add_argument("--result",       default=None,
                        help="Path to result_*.json (auto-detected if omitted).")
    args = parser.parse_args()

    if args.run:
        run_folder = args.run
    else:
        print("=" * 60)
        print("  VIDEOATLAS — Run Visualizer")
        print("=" * 60)

        results_dir = Path("results")
        if results_dir.is_dir():
            runs = sorted([d for d in results_dir.iterdir()
                           if d.is_dir() and d.name.endswith("_images")])
            if runs:
                print("\nAvailable runs:")
                for i, r in enumerate(runs):
                    n_imgs = len(list(r.glob("*.jpg")))
                    mtime  = max((f.stat().st_mtime for f in r.glob("*.jpg")), default=0)
                    age    = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(mtime))
                    print(f"  [{i}] {r.name}  ({n_imgs} images, last saved {age})")
                choice = input("\nEnter run number or full path: ").strip()
                if choice.isdigit() and int(choice) < len(runs):
                    run_folder = str(runs[int(choice)])
                else:
                    run_folder = choice
            else:
                run_folder = input("\nRun images folder path: ").strip()
        else:
            run_folder = input("\nRun images folder path: ").strip()

        fps_input = input("FPS (default 1.5 — lower is slower, e.g. 0.5): ").strip()
        args.fps  = float(fps_input) if fps_input else 1.5

    out = build_video(run_folder, args.output, args.fps, args.result)
    if out is None:
        sys.exit(1)


if __name__ == "__main__":
    main()

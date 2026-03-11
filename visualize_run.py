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
HEADER_HEIGHT    = 65    # Top bar: description + progress strip
TIMELINE_HEIGHT  = 28    # Bottom bar: phase scrubber
MAIN_WIDTH       = 960   # Left panel (main frame)
SCRATCHPAD_WIDTH = 560   # Right panel (question + evidence log)
OUTPUT_WIDTH     = MAIN_WIDTH + SCRATCHPAD_WIDTH   # 1520


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

# Plain-English explanation of each frame type — shown in the header bar.
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

# RGB colors — used directly in PIL and converted to BGR when applied to
# numpy arrays.  All three uses (header stripe, scrubber, progress bar) now
# pull from the same source so the palette is coherent.
CATEGORY_COLORS = {
    "global":     (80,  180, 120),
    "dfs_masked": (160, 130, 250),
    "dfs_uncert": (140, 100, 220),
    "worker":     (220, 160,  90),
    "bfs_masked": (250, 180,  60),
    "bfs_uncert": (230, 140,  50),
    "bfsworker":  (230, 200, 160),
    "zoom":       (60,  200, 220),
    "navgrid":    (130, 150, 230),
    "scratchpad": (100, 190, 150),
    "unknown":    (100, 100, 100),
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
# FONT & TEXT HELPERS
# ==========================================
def _get_font(size):
    for font_path in [
        "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    return ImageFont.load_default()


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


# ── UI Colors (PIL RGB) ────────────────────────────────────
_SCRATCHPAD_BG          = (30, 32, 36)
_SCRATCHPAD_TITLE_BG    = (42, 45, 52)
_SCRATCHPAD_LABEL_COLOR = (240, 240, 245)

_QUESTION_BOX_BG        = (35, 38, 50)
_QUESTION_BOX_BORDER    = (80, 100, 160)
_QUESTION_LABEL_FG      = (120, 140, 200)
_QUESTION_TEXT_FG       = (210, 215, 235)

_ANSWER_BOX_BG          = (40, 55, 85)
_ANSWER_BOX_BORDER      = (100, 150, 255)
_ANSWER_LABEL_FG        = (180, 200, 255)
_ANSWER_TEXT_FG         = (255, 255, 255)

_REASONING_FG           = (220, 220, 225)
_REASONING_TITLE        = (130, 200, 250)
_REASONING_NEW_BG       = (45,  55,  70)   # Subtle highlight for newly added items
_REASONING_NEW_TITLE    = (160, 220, 255)  # Brighter blue for newly added item headers
_DIVIDER_COLOR          = (65,  70,  80)
_TRUNCATED_FG           = (100, 110, 130)  # Muted indicator for items scrolled off


def build_scratchpad_panel(scratchpad_img_path, prev_sp_path,
                           panel_w, panel_h, result=None, question=None):
    """
    Build a fixed-size scratchpad panel (numpy BGR array).

    Layout (top → bottom):
      ┌─────────────────────────┐
      │  QUESTION box (always)  │  pinned from frame 1
      ├─────────────────────────┤
      │  FINAL CONCLUSION box   │  only after AI has concluded
      ├─────────────────────────┤
      │  title bar (36 px)      │
      ├─────────────────────────┤
      │  per-item reasoning     │  new items highlighted; truncation indicator shown
      └─────────────────────────┘
    """
    panel = np.full((panel_h, panel_w, 3), _SCRATCHPAD_BG, dtype=np.uint8)
    pil   = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw  = ImageDraw.Draw(pil)

    PAD          = 12
    font_title   = _get_font(18)
    font_item    = _get_font(14)
    font_q_label = _get_font(11)
    font_q_text  = _get_font(14)

    y_top = 0

    # ── QUESTION box — always shown when a question is available ──────
    if question:
        q_lines = _wrap_text(question, font_q_text, panel_w - PAD * 2, draw)
        box_h   = PAD + 16 + len(q_lines) * 18 + PAD
        draw.rectangle([(0, y_top), (panel_w - 1, y_top + box_h - 1)],
                        fill=_QUESTION_BOX_BG)
        draw.line([(0, y_top + box_h - 1), (panel_w - 1, y_top + box_h - 1)],
                   fill=_QUESTION_BOX_BORDER, width=2)
        draw.text((PAD, y_top + PAD), "QUESTION", font=font_q_label,
                   fill=_QUESTION_LABEL_FG)
        y = y_top + PAD + 16
        for ql in q_lines:
            draw.text((PAD, y), ql, font=font_q_text, fill=_QUESTION_TEXT_FG)
            y += 18
        y_top += box_h

    # ── FINAL CONCLUSION box — shown only after the last scratchpad ───
    if result is not None:
        font_ans_lbl  = _get_font(11)
        font_ans_main = _get_font(20)
        answer    = result.get("predicted_answer", "?")
        ans_lines = _wrap_text(f"Answer: {answer}", font_ans_main, panel_w - PAD * 2, draw)
        box_h     = PAD + 16 + 6 + len(ans_lines) * 26 + PAD

        draw.rectangle([(0, y_top), (panel_w - 1, y_top + box_h - 1)],
                        fill=_ANSWER_BOX_BG)
        draw.text((PAD, y_top + PAD), "FINAL CONCLUSION", font=font_ans_lbl,
                   fill=_ANSWER_LABEL_FG)
        y = y_top + PAD + 18
        draw.line([(PAD, y), (panel_w - PAD, y)], fill=_ANSWER_BOX_BORDER, width=1)
        y += 6
        for line in ans_lines:
            draw.text((PAD, y), line, font=font_ans_main, fill=_ANSWER_TEXT_FG)
            y += 26
        draw.line([(0, y_top + box_h), (panel_w - 1, y_top + box_h)],
                   fill=_ANSWER_BOX_BORDER, width=2)
        y_top += box_h + 2

    # ── Title bar ──────────────────────────────────────────────────────
    title_h = 36
    draw.rectangle([(0, y_top), (panel_w - 1, y_top + title_h - 1)],
                    fill=_SCRATCHPAD_TITLE_BG)

    if scratchpad_img_path is None:
        draw.text((PAD, y_top + 9), "Evidence Log  (collecting…)",
                   font=font_title, fill=(130, 130, 140))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    m       = re.search(r'scratchpad_(\d+)items', Path(scratchpad_img_path).stem)
    n_items = m.group(1) if m else "?"
    draw.text((PAD, y_top + 8), f"Evidence Log  —  {n_items} items",
               font=font_title, fill=_SCRATCHPAD_LABEL_COLOR)
    panel = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # ── Reasoning items ────────────────────────────────────────────────
    reasoning_items = _load_reasoning_for_scratchpad(scratchpad_img_path)
    prev_items      = _load_reasoning_for_scratchpad(prev_sp_path)
    new_count       = len(reasoning_items) - len(prev_items)   # items added since last scratchpad
    reasoning_y     = y_top + title_h + 10

    if not reasoning_items:
        return panel

    pil2  = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw2 = ImageDraw.Draw(pil2)

    ITEM_MARGIN  = 16
    LINE_H       = 19
    font_ev_hdr  = _get_font(15)

    all_blocks = []
    for idx, item in enumerate(reasoning_items):
        is_new      = new_count > 0 and idx >= len(reasoning_items) - new_count
        letter      = item.get("letter", "?")
        t           = item.get("time", 0.0)
        desc        = item.get("description", "")
        header_text = f"Event {letter} • {_format_time(t)}"
        desc_lines  = _wrap_text(desc, font_item, panel_w - PAD * 2 - 4, draw2)

        title_color = _REASONING_NEW_TITLE if is_new else _REASONING_TITLE
        block_lines = [(header_text, title_color, font_ev_hdr, is_new)]
        for dl in desc_lines:
            block_lines.append((dl, _REASONING_FG, font_item, is_new))

        block_h = (len(block_lines) * LINE_H) + ITEM_MARGIN
        all_blocks.append((block_lines, block_h, is_new))

    # Clip to available height, keeping newest items; emit truncation indicator
    avail_h     = panel_h - reasoning_y - 10
    total_h     = sum(bh for _, bh, _ in all_blocks)
    start_block = 0

    if total_h > avail_h:
        running = 0
        for i, (_, bh, _) in enumerate(all_blocks):
            running += bh
            if running >= total_h - avail_h:
                start_block = i
                break

    if start_block > 0:
        font_trunc  = _get_font(11)
        skipped_txt = f"↑  {start_block} earlier item{'s' if start_block != 1 else ''} not shown"
        draw2.text((PAD, reasoning_y), skipped_txt, font=font_trunc, fill=_TRUNCATED_FG)
        reasoning_y += 18

    cur_y = reasoning_y
    for block_lines, block_h, is_new in all_blocks[start_block:]:
        if cur_y + block_h > panel_h - 2:
            break
        if is_new:
            draw2.rectangle(
                [(2, cur_y - 2), (panel_w - 3, cur_y + block_h - ITEM_MARGIN + 2)],
                fill=_REASONING_NEW_BG,
            )
        for line_text, line_color, line_font, new_flag in block_lines:
            draw2.text((PAD + (4 if new_flag else 0), cur_y), line_text,
                        font=line_font, fill=line_color)
            cur_y += LINE_H
        draw2.line([(PAD, cur_y), (panel_w - PAD, cur_y)], fill=_DIVIDER_COLOR, width=1)
        cur_y += ITEM_MARGIN

    return cv2.cvtColor(np.array(pil2), cv2.COLOR_RGB2BGR)


# ==========================================
# HEADER
# ==========================================
def make_header(label, category, frame_num, total_frames):
    color_rgb = CATEGORY_COLORS.get(category, (80, 80, 80))

    header = np.full((HEADER_HEIGHT, OUTPUT_WIDTH, 3), (22, 24, 28), dtype=np.uint8)
    # Accent stripe — RGB→BGR for the numpy array
    header[:, :10, :] = _rgb_to_bgr(color_rgb)

    pil  = Image.fromarray(cv2.cvtColor(header, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    font_big   = _get_font(22)
    font_small = _get_font(13)

    desc = CATEGORY_DESCRIPTIONS.get(category, label)
    draw.text((22, 14), desc, font=font_big, fill=(245, 245, 250))

    frame_text = f"Frame {frame_num} / {total_frames}"
    w_frame    = draw.textlength(frame_text, font=font_small)
    draw.text((MAIN_WIDTH - w_frame - 20, 18), frame_text,
               font=font_small, fill=(100, 110, 120))

    # Progress bar — thin strip at the bottom of the header
    bar_x1, bar_x2 = 22, OUTPUT_WIDTH - 22
    bar_y, bar_h   = HEADER_HEIGHT - 11, 5
    draw.rectangle([(bar_x1, bar_y), (bar_x2, bar_y + bar_h)], fill=(45, 50, 60))
    fill_x = bar_x1 + int((bar_x2 - bar_x1) * frame_num / total_frames)
    if fill_x > bar_x1:
        draw.rectangle([(bar_x1, bar_y), (fill_x, bar_y + bar_h)], fill=color_rgb)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ==========================================
# TIMELINE SCRUBBER
# ==========================================
def build_timeline_scrubber(categories_all, current_idx, width, height):
    """
    Full-width bar at the bottom of the frame.  Each segment represents one
    frame, colored by its activity category.  Past frames are dimmed; the
    current frame is marked with a bright vertical tick.  Viewers can see at
    a glance where in the overall exploration process each frame falls.
    """
    bar   = np.full((height, width, 3), (20, 22, 26), dtype=np.uint8)
    total = len(categories_all)
    if total == 0:
        return bar

    seg_w = width / total
    for i, cat in enumerate(categories_all):
        color = CATEGORY_COLORS.get(cat, (80, 80, 80))
        x1    = int(i * seg_w)
        x2    = max(x1 + 1, int((i + 1) * seg_w))
        if i < current_idx:
            c_bgr = (int(color[2] * 0.4), int(color[1] * 0.4), int(color[0] * 0.4))
        else:
            c_bgr = _rgb_to_bgr(color)
        bar[5:height - 5, x1:x2] = c_bgr

    # Current-position marker
    cx = int((current_idx + 0.5) * seg_w)
    bar[:, max(0, cx - 1):min(width, cx + 2)] = (255, 255, 255)

    # "TIMELINE" label on the left
    pil  = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((6, (height - 11) // 2), "TIMELINE", font=_get_font(10),
               fill=(140, 150, 165))

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
    # ---- Main frame ----
    img = cv2.imread(str(img_path))
    if img is None:
        img = np.zeros((target_content_h, MAIN_WIDTH, 3), dtype=np.uint8)
        cv2.putText(img, f"Could not load: {img_path.name}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        h, w  = img.shape[:2]
        new_h = int(h * MAIN_WIDTH / w)
        img   = cv2.resize(img, (MAIN_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        if img.shape[0] < target_content_h:
            pad = np.zeros((target_content_h - img.shape[0], MAIN_WIDTH, 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        elif img.shape[0] > target_content_h:
            img = img[:target_content_h, :, :]

    # ---- Scratchpad panel ----
    sp_panel = build_scratchpad_panel(
        scratchpad_img_path, prev_sp_path,
        SCRATCHPAD_WIDTH, target_content_h,
        result=result, question=question,
    )

    # ---- Vertical divider ----
    divider = np.full((target_content_h, 2, 3), (60, 60, 60), dtype=np.uint8)

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
    result = subprocess.run(cmd, capture_output=True, text=True)
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
    target_content_h = int(np.median(sample_heights)) if sample_heights else 720
    total_h          = target_content_h + HEADER_HEIGHT + TIMELINE_HEIGHT

    tmp_path = output_path.replace(".mp4", "_raw.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(tmp_path, fourcc, fps, (OUTPUT_WIDTH, total_h))

    if not writer.isOpened():
        print(f"[!] Could not open VideoWriter. Try installing opencv-python.")
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
        print(f"\n✓  Saved (H.264): {output_path}")
    else:
        import shutil
        shutil.move(tmp_path, output_path)
        print(f"\n✓  Saved (mp4v fallback): {output_path}")
        print("   Install ffmpeg for full WhatsApp/phone compatibility.")

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
        print("  VIDEO EXPLORER — Run Visualizer")
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

"""
visualize_run.py — Turn a previous Video Explorer run into a replay video.

Frames are shown in TRUE RUN ORDER using the leading sequence number in
filenames, so the video replays exactly what the system did, step by step.

The right-hand side shows a LIVE SCRATCHPAD PANEL that updates whenever
a new scratchpad_* image is detected in the run folder — giving a real-time
view of the evidence collected so far.

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
import sys
import time as _time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ==========================================
# LAYOUT CONSTANTS
# ==========================================
HEADER_HEIGHT    = 90   # extra height to show the plain-English description line
MAIN_WIDTH       = 1024   # left panel (main frame)
SCRATCHPAD_WIDTH = 520    # right panel (live scratchpad + reasoning)
OUTPUT_WIDTH     = MAIN_WIDTH + SCRATCHPAD_WIDTH


# ==========================================
# FRAME CLASSIFICATION  (label + colour only)
# ==========================================
CATEGORIES = [
    (r"^global_grid",                    "global",      "Global Grid — Master Overview"),
    (r"^DFS_round(\d+)_masked_grid",     "dfs_masked",  "DFS Round {1} — Masked Grid (Master Probe)"),
    (r"^DFS_round(\d+)_uncertainty",     "dfs_uncert",  "DFS Round {1} — Uncertainty Analysis Grid"),
    (r"^W(\d+)_C(\d+)_step(\d+)",        "worker",      "Worker {1} · Cell {2} · Step {3}"),
    (r"^BFS_batch(\d+)_masked_grid",     "bfs_masked",  "BFS Batch {1} — Masked Grid (Master Probe)"),
    (r"^BFS_batch(\d+)_uncertainty",     "bfs_uncert",  "BFS Batch {1} — Uncertainty Analysis Grid"),
    (r"^BFSW(\d+)_depth(\d+)_step(\d+)","bfsworker",   "BFS Worker {1} · Depth {2} · Step {3}"),
    (r"^zoom_(.+?)_\d{3}$",             "zoom",        "Zoom @ {1}"),
    (r"^grid_c(.+?)s_span(.+?)s",        "navgrid",     "Navigator Grid  center={1}s  span={2}s"),
    (r"^scratchpad_(\d+)items",          "scratchpad",  "Scratchpad Evidence Grid  ({1} items)"),
]

# Plain-English explanation of each frame type — shown in the header bar so a
# viewer who has never seen the tool before can follow along.
CATEGORY_DESCRIPTIONS = {
    "global":     "Getting a bird's-eye view of the entire video",
    "dfs_masked": "Deciding which parts of the video still need closer inspection",
    "dfs_uncert": "Scoring how confident the AI is across each region of the video",
    "worker":     "A worker agent is zooming in on a specific moment to examine it closely",
    "bfs_masked": "Marking which areas of the video have already been explored",
    "bfs_uncert": "Re-evaluating confidence scores across all explored regions",
    "bfsworker":  "A worker agent is drilling deeper into a promising region of the video",
    "zoom":       "Examining a specific timestamp up close",
    "navgrid":    "Navigating the video timeline to choose the next region to explore",
    "scratchpad": "Reviewing all the evidence collected so far",
    "unknown":    "Processing…",
}

CATEGORY_COLORS = {
    "global":     (34,  139,  34),
    "dfs_masked": (180, 105, 255),
    "dfs_uncert": (130,  60, 200),
    "worker":     (205, 133,  63),
    "bfs_masked": (255, 165,   0),
    "bfs_uncert": (200, 120,   0),
    "bfsworker":  (210, 180, 140),
    "zoom":       (0,   200, 200),
    "navgrid":    (100, 100, 200),
    "scratchpad": (60,  179, 113),
    "unknown":    (80,   80,  80),
}


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
# FONT HELPER
# ==========================================
def _get_font(size):
    for font_path in [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    return ImageFont.load_default()



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
    p    = Path(scratchpad_img_path)
    json_path = p.parent / (p.stem + "_reasoning.json")
    if not json_path.exists():
        return []
    try:
        with open(json_path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

_SCRATCHPAD_LABEL_COLOR = (200, 230, 200)
_SCRATCHPAD_BG          = (20, 20, 20)
_SCRATCHPAD_TITLE_BG    = (40, 80, 40)

# ── Final-answer box colours (PIL RGB) ────────────────────────────
_ANSWER_BOX_BG      = (12, 55, 12)     # dark green background
_ANSWER_BOX_BORDER  = (60, 200, 60)    # bright green border line
_ANSWER_LABEL_FG    = (100, 255, 120)  # "FINAL ANSWER" label text
_ANSWER_TEXT_FG     = (255, 255, 200)  # answer body text


_REASONING_BG    = (30, 30, 30)
_REASONING_FG    = (210, 210, 210)
_REASONING_HL    = (130, 230, 130)   # letter + timestamp highlight
_CONF_HIGH       = (80,  200, 80)    # confidence ≥ 0.85
_CONF_MED        = (200, 200, 80)    # confidence ≥ 0.70
_CONF_LOW        = (200, 100, 80)    # confidence < 0.70
_DIVIDER_COLOR   = (60,  60,  60)


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


def build_scratchpad_panel(scratchpad_img_path, panel_w, panel_h, result=None):
    """
    Build a fixed-size scratchpad panel (numpy BGR array).

    Layout (top → bottom):
      ┌─────────────────────────┐
      │  FINAL ANSWER box       │  (only shown once the AI has concluded)
      ├─────────────────────────┤
      │  title bar (30 px)      │
      ├─────────────────────────┤
      │  scratchpad grid image  │  (≤ 40 % of remaining panel height)
      ├─────────────────────────┤
      │  per-item reasoning     │  (remaining space, scrolled to latest)
      └─────────────────────────┘
    """
    panel = np.full((panel_h, panel_w, 3), _SCRATCHPAD_BG, dtype=np.uint8)

    pil  = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font_title  = _get_font(17)
    font_item   = _get_font(13)
    font_small  = _get_font(11)

    y_top = 0   # moves down as we stack sections

    # ── FINAL ANSWER box (shown only after the last scratchpad frame) ──
    if result is not None:
        font_ans_lbl  = _get_font(13)
        font_ans_main = _get_font(17)
        choice = result.get("predicted_choice", "?")
        answer = result.get("predicted_answer", "?")
        # Wrap the full answer — no truncation
        ans_lines = _wrap_text(f"[{choice}]  {answer}", font_ans_main, panel_w - 14, draw)
        PAD   = 6
        box_h = PAD + 16 + 3 + len(ans_lines) * 21 + PAD
        draw.rectangle([(0, 0), (panel_w - 1, box_h - 1)], fill=_ANSWER_BOX_BG)
        draw.text((PAD, PAD), "FINAL ANSWER", font=font_ans_lbl, fill=_ANSWER_LABEL_FG)
        ay = PAD + 16 + 3
        for line in ans_lines:
            draw.text((PAD, ay), line, font=font_ans_main, fill=_ANSWER_TEXT_FG)
            ay += 21
        draw.line([(0, box_h), (panel_w - 1, box_h)], fill=_ANSWER_BOX_BORDER, width=2)
        y_top = box_h + 2

    # ── title bar ──────────────────────────────────────────────────
    title_h = 30
    draw.rectangle([(0, y_top), (panel_w - 1, y_top + title_h - 1)],
                   fill=_SCRATCHPAD_TITLE_BG)

    if scratchpad_img_path is None:
        draw.text((8, y_top + 7), "SCRATCHPAD  (no evidence yet)",
                  font=font_title, fill=_SCRATCHPAD_LABEL_COLOR)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    m       = re.search(r'scratchpad_(\d+)items', Path(scratchpad_img_path).stem)
    n_items = m.group(1) if m else "?"
    draw.text((8, y_top + 7), f"SCRATCHPAD  ({n_items} evidence items)",
              font=font_title, fill=_SCRATCHPAD_LABEL_COLOR)
    panel = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # ── scratchpad grid image ──────────────────────────────────────
    sp_img = cv2.imread(str(scratchpad_img_path))
    img_section_h = 0
    img_y_start   = y_top + title_h
    if sp_img is not None:
        max_img_h  = int(panel_h * 0.40)
        h, w       = sp_img.shape[:2]
        scale      = min(panel_w / w, max_img_h / h)
        new_w      = int(w * scale)
        new_h      = int(h * scale)
        sp_resized = cv2.resize(sp_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x_off      = (panel_w - new_w) // 2
        y_off      = img_y_start
        if y_off + new_h <= panel_h:
            panel[y_off:y_off + new_h, x_off:x_off + new_w] = sp_resized
            img_section_h = new_h

    # ── reasoning section ──────────────────────────────────────────
    reasoning_items = _load_reasoning_for_scratchpad(scratchpad_img_path)
    reasoning_y     = img_y_start + img_section_h + 4

    # thin divider between image and text
    if img_section_h > 0 and reasoning_items:
        panel[reasoning_y - 2:reasoning_y, :] = _DIVIDER_COLOR

    if not reasoning_items:
        return panel

    # Convert panel to PIL for text rendering
    pil2  = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw2 = ImageDraw.Draw(pil2)

    # Build all lines first so we can clip to the bottom (show most recent)
    ITEM_MARGIN  = 6    # px between items
    LINE_H_ITEM  = 16   # px per line for description
    LINE_H_SMALL = 13   # px per line for subtitle
    PAD_X        = 6

    all_blocks = []   # list of (lines_to_draw: [(text, color, font)], block_h)

    for item in reasoning_items:
        letter  = item.get("letter", "?")
        t       = item.get("time", 0.0)
        conf    = item.get("confidence", 0.0)
        desc    = item.get("description", "")
        sub     = item.get("subtitle", "")

        conf_color = _CONF_HIGH if conf >= 0.85 else (_CONF_MED if conf >= 0.70 else _CONF_LOW)

        header_text  = f"[{letter}] @{t:.1f}s  conf={conf:.2f}"
        desc_lines   = _wrap_text(desc, font_item, panel_w - PAD_X * 2, draw2)
        sub_lines    = _wrap_text(f"♪ {sub}", font_small, panel_w - PAD_X * 2, draw2) if sub else []

        block_lines = [(header_text, conf_color, font_item)]
        for dl in desc_lines:
            block_lines.append((dl, _REASONING_FG, font_item))
        for sl in sub_lines:
            block_lines.append((sl, (140, 160, 200), font_small))

        block_h = (len(block_lines) * LINE_H_ITEM) + ITEM_MARGIN
        all_blocks.append((block_lines, block_h))

    # Measure total height; if overflow, skip oldest blocks to show recent ones
    avail_h     = panel_h - reasoning_y - 4
    total_h     = sum(bh for _, bh in all_blocks)
    start_block = 0
    if total_h > avail_h:
        running = 0
        for i, (_, bh) in enumerate(all_blocks):
            running += bh
            if running >= total_h - avail_h:
                start_block = i
                break

    cur_y = reasoning_y + 4
    for block_lines, block_h in all_blocks[start_block:]:
        if cur_y + block_h > panel_h - 2:
            break
        for line_text, line_color, line_font in block_lines:
            draw2.text((PAD_X, cur_y), line_text, font=line_font, fill=line_color)
            cur_y += LINE_H_ITEM
        # small divider between items
        draw2.line([(PAD_X, cur_y), (panel_w - PAD_X, cur_y)], fill=_DIVIDER_COLOR, width=1)
        cur_y += ITEM_MARGIN

    return cv2.cvtColor(np.array(pil2), cv2.COLOR_RGB2BGR)


# ==========================================
# HEADER + FRAME ASSEMBLY
# ==========================================
def make_header(label, category, frame_num, total_frames):
    color_bgr = CATEGORY_COLORS.get(category, (80, 80, 80))
    header    = np.full((HEADER_HEIGHT, OUTPUT_WIDTH, 3), color_bgr, dtype=np.uint8)

    pil  = Image.fromarray(cv2.cvtColor(header, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font_large = _get_font(32)
    font_small = _get_font(20)
    font_desc  = _get_font(15)

    # Line 1 — frame type name (technical label)
    draw.text((12, 4),  label,                                 font=font_large, fill=(255, 255, 255))
    # Line 2 — frame counter
    draw.text((12, 38), f"Frame {frame_num} / {total_frames}", font=font_small,  fill=(220, 220, 220))
    # Line 3 — plain-English description of what the AI is doing right now
    desc = CATEGORY_DESCRIPTIONS.get(category, "")
    if desc:
        draw.text((12, 62), desc, font=font_desc, fill=(230, 230, 180))

    # Scratchpad panel title in header (right side)
    draw.text((MAIN_WIDTH + 8, 35), "◀  LIVE SCRATCHPAD  ▶",
              font=font_small, fill=(180, 255, 180))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def prepare_frame(img_path, label, category, frame_num, total_frames,
                  target_content_h, scratchpad_img_path, result=None):
    """
    Build one output frame: header | (main_frame | divider | scratchpad_panel)

    `result` is the full result dict; when provided the scratchpad panel shows
    the FINAL ANSWER box (only passed in once the last scratchpad has appeared).
    """
    # ---- Main frame (left panel) ----
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

    # ---- Scratchpad panel (right panel) ----
    sp_panel = build_scratchpad_panel(scratchpad_img_path, SCRATCHPAD_WIDTH, target_content_h,
                                      result=result)

    # ---- Vertical divider (2px) ----
    divider = np.full((target_content_h, 2, 3), (60, 60, 60), dtype=np.uint8)

    # ---- Combine left + divider + right ----
    content_row = np.hstack([img, divider, sp_panel])

    # Guard against ±1px rounding
    if content_row.shape[1] != OUTPUT_WIDTH:
        content_row = cv2.resize(content_row, (OUTPUT_WIDTH, target_content_h))

    header = make_header(label, category, frame_num, total_frames)
    return np.vstack([header, content_row])


# ==========================================
# SCRATCHPAD TIMELINE BUILDER
# ==========================================
def build_scratchpad_timeline(classified):
    """
    For each frame index i, return the path of the most recent scratchpad_*
    image seen at or before frame i. Returns a list of (Path | None).
    """
    timeline = []
    current  = None
    for category, _label, img_path in classified:
        if category == "scratchpad":
            current = img_path
        timeline.append(current)
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
    import subprocess
    # tpad=stop_mode=clone:stop_duration=N freezes the last frame for N seconds
    vf = f"tpad=stop_mode=clone:stop_duration={end_hold_seconds}"
    cmd = (
        f'ffmpeg -y -i "{input_path}" '
        f'-vf "{vf}" '
        f'-c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p '
        f'-an -movflags +faststart '
        f'"{output_path}"'
    )
    print(f"  Re-encoding to H.264 with ffmpeg (+ {end_hold_seconds}s end freeze)...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
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

    # Sort by leading sequence number
    def _seq(p):
        m = re.match(r'^(\d+)_', p.name)
        return int(m.group(1)) if m else 0

    all_images.sort(key=_seq)
    print("  Sorted by sequence number (true run order).")

    # Classify each image
    classified = []
    for img_path in all_images:
        bare_name = re.sub(r'^\d+_', '', img_path.name)
        category, label = classify(bare_name)
        classified.append((category, label, img_path))

    # Build the per-frame scratchpad timeline
    sp_timeline = build_scratchpad_timeline(classified)

    # Load result JSON
    if result_json_path:
        with open(result_json_path) as f:
            result = json.load(f)
    else:
        result = load_result(run_folder)

    if result:
        print(f"  Result: [{result.get('predicted_choice')}] "
              f"{result.get('predicted_answer', '')[:60]}")
    else:
        print("  No result JSON found — answer overlay disabled.")

    # Find the index of the very last scratchpad frame — the answer box will
    # only appear on that frame and all frames that come after it.
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

    # Determine consistent target content height from sample images
    sample_heights = []
    for _, _, img_path in classified[:min(20, total)]:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            sample_heights.append(int(h * MAIN_WIDTH / w))
    target_content_h = int(np.median(sample_heights)) if sample_heights else 720
    total_h = target_content_h + HEADER_HEIGHT

    tmp_path = output_path.replace(".mp4", "_raw.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(tmp_path, fourcc, fps, (OUTPUT_WIDTH, total_h))

    if not writer.isOpened():
        print(f"[!] Could not open VideoWriter. Try installing opencv-python.")
        return None

    for i, (category, label, img_path) in enumerate(classified, 1):
        sp_path = sp_timeline[i - 1]
        # Reveal the answer only once we've passed the final scratchpad frame.
        # If there is no scratchpad at all, show it from the very first frame.
        result_for_frame = result if (last_sp_idx is None or (i - 1) >= last_sp_idx) else None
        frame   = prepare_frame(
            img_path, label, category, i, total,
            target_content_h, sp_path, result_for_frame
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
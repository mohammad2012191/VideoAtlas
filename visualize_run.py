"""
visualize_run.py — Turn a previous Video Explorer run into a replay video.

Frames are shown in TRUE RUN ORDER using file modification timestamps,
so the video replays exactly what the system did, step by step.

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
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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

CATEGORY_COLORS = {
    "global":     (34,  139,  34),   # forest green
    "dfs_masked": (180, 105, 255),   # pink/purple
    "dfs_uncert": (130,  60, 200),   # deep purple
    "worker":     (205, 133,  63),   # peru brown
    "bfs_masked": (255, 165,   0),   # orange
    "bfs_uncert": (200, 120,   0),   # dark orange
    "bfsworker":  (210, 180, 140),   # tan
    "zoom":       (0,   200, 200),   # cyan
    "navgrid":    (100, 100, 200),   # muted blue
    "scratchpad": (60,  179, 113),   # medium sea green
    "unknown":    (80,   80,  80),   # grey
}


def classify(filename):
    """Return (category, label) for a filename."""
    stem = Path(filename).stem
    # Strip trailing _NNN counter to get the base name
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
# FRAME ANNOTATION
# ==========================================
HEADER_HEIGHT = 72
OUTPUT_WIDTH  = 1280


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


def make_header(label, category, frame_num, total_frames, result_info, width):
    color_bgr = CATEGORY_COLORS.get(category, (80, 80, 80))
    header    = np.full((HEADER_HEIGHT, width, 3), color_bgr, dtype=np.uint8)

    pil  = Image.fromarray(cv2.cvtColor(header, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font_large = _get_font(22)
    font_small = _get_font(15)

    # Main label
    draw.text((12, 6),  label,                        font=font_large, fill=(255, 255, 255))
    # Frame counter
    draw.text((12, 38), f"Frame {frame_num} / {total_frames}", font=font_small, fill=(220, 220, 220))
    # Answer overlay (far right)
    if result_info:
        bbox   = draw.textbbox((0, 0), result_info, font=font_small)
        text_w = bbox[2] - bbox[0]
        draw.text((width - text_w - 12, 6), result_info, font=font_small, fill=(255, 255, 180))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def prepare_frame(img_path, label, category, frame_num, total_frames, result_info, target_h):
    img = cv2.imread(str(img_path))
    if img is None:
        img = np.zeros((400, OUTPUT_WIDTH, 3), dtype=np.uint8)
        cv2.putText(img, f"Could not load: {img_path.name}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Resize width to OUTPUT_WIDTH, keep aspect ratio
    h, w  = img.shape[:2]
    new_h = int(h * OUTPUT_WIDTH / w)
    img   = cv2.resize(img, (OUTPUT_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    # Pad or crop to target_h so all frames are the same height
    if img.shape[0] < target_h:
        pad = np.zeros((target_h - img.shape[0], OUTPUT_WIDTH, 3), dtype=np.uint8)
        img = np.vstack([img, pad])
    elif img.shape[0] > target_h:
        img = img[:target_h, :, :]

    header = make_header(label, category, frame_num, total_frames, result_info, OUTPUT_WIDTH)
    return np.vstack([header, img])


# ==========================================
# RESULT JSON LOADER
# ==========================================
def load_result(run_folder):
    run_folder  = Path(run_folder)
    parent      = run_folder.parent
    stem        = run_folder.stem                              # run_20240101_120000_images
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
    q      = result.get("question", "")[:45]
    return f"Q: {q}...  →  [{choice}] {answer[:35]}"


# ==========================================
# BUILD VIDEO
# ==========================================
def build_video(run_folder, output_path, fps, result_json_path=None):
    run_folder = Path(run_folder)

    if not run_folder.is_dir():
        print(f"[!] Folder not found: {run_folder}")
        sys.exit(1)

    all_images = list(run_folder.glob("*.jpg"))
    if not all_images:
        print(f"[!] No .jpg images found in: {run_folder}")
        sys.exit(1)

    print(f"\nFound {len(all_images)} images in: {run_folder}")

    # ---- Sort by the leading sequence number in the filename (true run order) ----
    def _seq(p):
        m = re.match(r'^(\d+)_', p.name)
        return int(m.group(1)) if m else 0

    all_images.sort(key=_seq)
    print("  Sorted by sequence number (true run order).")

    # Classify each image (for label + colour only — order already fixed)
    classified = []
    for img_path in all_images:
        # Strip leading NNNN_ prefix before classifying
        bare_name = re.sub(r'^\d+_', '', img_path.name)
        category, label = classify(bare_name)
        classified.append((category, label, img_path))

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

    result_info = format_result_info(result)

    if output_path is None:
        output_path = str(run_folder / "replay.mp4")

    total = len(classified)
    print(f"\nBuilding video: {total} frames @ {fps} fps  →  {output_path}")

    # Determine a consistent target content height from the median image height
    sample_heights = []
    for _, _, img_path in classified[:min(20, total)]:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            sample_heights.append(int(h * OUTPUT_WIDTH / w))
    target_content_h = int(np.median(sample_heights)) if sample_heights else 720
    total_h = target_content_h + HEADER_HEIGHT

    # Write frames to a temporary raw file first, then re-encode to H.264 with ffmpeg.
    # mp4v (OpenCV default) is not compatible with WhatsApp/phones — H.264 is required.
    tmp_path = output_path.replace(".mp4", "_raw.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (OUTPUT_WIDTH, total_h))

    if not writer.isOpened():
        print(f"[!] Could not open VideoWriter. Try installing opencv-python.")
        sys.exit(1)

    for i, (category, label, img_path) in enumerate(classified, 1):
        frame = prepare_frame(img_path, label, category, i, total, result_info, target_content_h)
        writer.write(frame)

        if i % 10 == 0 or i == total:
            print(f"  [{i:4d}/{total}]  {label[:70]}")

    writer.release()

    # Re-encode to H.264 using ffmpeg for WhatsApp / phone compatibility
    duration = total / fps
    ffmpeg_ok = _reencode_h264(tmp_path, output_path)

    if ffmpeg_ok:
        os.remove(tmp_path)
        print(f"\n✓  Saved (H.264): {output_path}")
    else:
        import shutil
        shutil.move(tmp_path, output_path)
        print(f"\n✓  Saved (mp4v fallback — may not play on WhatsApp): {output_path}")
        print("   Install ffmpeg for full WhatsApp/phone compatibility.")

    print(f"   {total} frames · {fps} fps · {duration:.1f}s duration")


def _reencode_h264(input_path, output_path):
    """
    Re-encode to H.264 + yuv420p using ffmpeg.
    Flags:
      -c:v libx264       H.264 — universally supported
      -preset fast       good speed/quality tradeoff
      -crf 23            quality (lower = better; 23 is a good default)
      -pix_fmt yuv420p   required for WhatsApp and most phones/players
      -an                no audio (we have none)
      -movflags +faststart  puts metadata at front for fast open/streaming
    Returns True on success, False if ffmpeg unavailable or fails.
    """
    import subprocess
    cmd = (
        f'ffmpeg -y -i "{input_path}" '
        f'-c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p '
        f'-an -movflags +faststart '
        f'"{output_path}"'
    )
    print("  Re-encoding to H.264 with ffmpeg...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [!] ffmpeg re-encode failed:\n{result.stderr[-400:]}")
        return False
    return True


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
                    import time as _time
                    age = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(mtime))
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

        fps_input  = input("FPS (default 1.5 — lower is slower, e.g. 0.5): ").strip()
        args.fps   = float(fps_input) if fps_input else 1.5

    build_video(run_folder, args.output, args.fps, args.result)


if __name__ == "__main__":
    main()
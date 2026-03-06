"""
navigator.py — Video frame extraction, grid generation, and subtitle reading.
"""

import cv2
import json
import os
import random
import numpy as np
from PIL import Image

from config import GRID_K
from memory import NegativeMemory, VisualScratchpad
from metrics import metrics
from logger import log, save_debug_image


# ==========================================
# SUBTITLE READER
# ==========================================
class SubtitleReader:
    def __init__(self, subtitle_path):
        self.subs = []
        if subtitle_path and os.path.exists(subtitle_path):
            with open(subtitle_path, 'r') as f:
                data = json.load(f)
            for item in data:
                text_content = item.get('text', item.get('content', item.get('line', '')))
                raw_start    = item.get('start', item.get('timestamp', [0, 0])[0])
                raw_end      = item.get('end',   item.get('timestamp', [0, 0])[1])
                self.subs.append({
                    'start': self._parse_time(raw_start),
                    'end':   self._parse_time(raw_end),
                    'text':  text_content
                })

    def _parse_time(self, time_val):
        if isinstance(time_val, (int, float)):
            return float(time_val)
        if isinstance(time_val, str):
            parts = time_val.split(':')
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
        return 0.0

    def get_text_for_interval(self, start, end):
        matches = [s for s in self.subs if s['start'] < end and s['end'] > start]
        if not matches:
            return ""
        if len(matches) <= 3:
            return " | ".join(m['text'] for m in matches)

        first    = matches[0]['text']
        last     = matches[-1]['text']
        mid_time = (start + end) / 2
        mid_sub  = min(matches, key=lambda s: abs((s['start'] + s['end']) / 2 - mid_time))
        mid_text = mid_sub['text']

        parts = [first]
        if mid_text != first:
            parts.append(mid_text)
        if last != parts[-1]:
            parts.append(last)
        return " | ".join(parts)


# ==========================================
# VISUAL NAVIGATOR
# ==========================================
class VisualNavigator:
    def __init__(self, video_path, sub_path, grid_k=GRID_K):
        self.cap          = cv2.VideoCapture(video_path)
        self.fps          = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration     = self.total_frames / self.fps
        self.k            = grid_k
        self.subs         = SubtitleReader(sub_path)
        self.negative_mem = NegativeMemory(self.duration)
        self.scratchpad   = VisualScratchpad()
        self.visit_queue  = []

    def get_frame(self, t):
        fid = int(t * self.fps)
        fid = max(0, min(fid, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = self.cap.read()
        metrics.add_frames(1)
        if not ret:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def generate_grid_view(self, center, span, random_sample=False):
        start = max(0, center - span / 2)
        end   = min(self.duration, center + span / 2)
        step  = (end - start) / (self.k * self.k)

        frames    = []
        cell_info = []

        for i in range(self.k * self.k):
            cell_start   = start + i * step
            cell_end     = cell_start + step
            sampled_time = random.uniform(cell_start, cell_end) if random_sample \
                           else (cell_start + cell_end) / 2
            sampled_time = max(0, min(self.duration, sampled_time))

            is_dead = self.negative_mem.is_dead_interval(cell_start, cell_end)

            img = self.get_frame(sampled_time)
            img = cv2.resize(img, (336, 336))
            if is_dead:
                img = (img * 0.0).astype(np.uint8)

            frames.append(img)
            sub_text = self.subs.get_text_for_interval(cell_start, cell_end) or "(No speech)"
            cell_info.append({
                "id":         i,
                "start":      cell_start,
                "end":        cell_end,
                "time_range": f"{cell_start:.1f}-{cell_end:.1f}s",
                "subtitles":  sub_text,
                "status":     "DEAD" if is_dead else "OPEN"
            })

        rows     = [np.hstack(frames[i:i + self.k]) for i in range(0, len(frames), self.k)]
        grid_img = np.vstack(rows)
        pil_grid = Image.fromarray(grid_img)
        save_debug_image(pil_grid, f"grid_c{center:.0f}s_span{span:.0f}s")
        return pil_grid, cell_info, start, end

    def get_full_frame(self, timestamp, duration=0.0):
        if duration > 0:
            step  = duration / 4
            times = [timestamp + i * step for i in range(5)]
        else:
            times = [timestamp]

        frames = []
        for t in times:
            t   = max(0, min(t, self.duration))
            img = self.get_frame(t)
            img = cv2.resize(img, (512, 512))
            frames.append(img)

        full_img = np.hstack(frames) if len(frames) > 1 else frames[0]
        sub_text = self.subs.get_text_for_interval(min(times), max(times) + 0.1)
        pil_full = Image.fromarray(full_img)
        save_debug_image(pil_full, f"zoom_{min(times):.1f}s")
        return pil_full, sub_text, min(times), max(times)


# ==========================================
# GRID UTILITY FUNCTIONS
# ==========================================
def build_context_str(cell_info):
    """Compact context: skip DEAD cells and cells with no speech."""
    lines = []
    for c in cell_info:
        if c['status'] == 'DEAD':
            continue
        sub = c['subtitles']
        if sub == "(No speech)":
            lines.append(f"- Cell {c['id']} ({c['time_range']})")
        else:
            lines.append(f"- Cell {c['id']} ({c['time_range']}): {sub}")
    return "\n".join(lines)


def blackout_dead_cells(grid_img, cell_info, grid_k=GRID_K, cell_px=336):
    """Return a copy of the grid image with DEAD cells blacked out."""
    arr = np.array(grid_img).copy()
    for ci in cell_info:
        if ci.get('status') == 'DEAD':
            row = ci['id'] // grid_k
            col = ci['id'] % grid_k
            arr[row * cell_px:(row + 1) * cell_px,
                col * cell_px:(col + 1) * cell_px] = 0
    return Image.fromarray(arr)


def build_progress_text(nav, explored_ranges, scratchpad):
    """Lightweight temporal overview: explored ranges + evidence timestamps + gaps."""
    duration      = nav.duration
    sorted_ranges = sorted(explored_ranges)
    merged        = []
    for s, e in sorted_ranges:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    explored_pct = sum(e - s for s, e in merged) / duration * 100 if duration > 0 else 0
    lines = [f"Video: {duration:.0f}s | Explored: {explored_pct:.0f}%"]

    if merged:
        lines.append("Explored: " + ", ".join(f"[{s:.0f}-{e:.0f}s]" for s, e in merged))

    gaps     = []
    prev_end = 0.0
    for s, e in merged:
        if s > prev_end + 5:
            gaps.append((prev_end, s))
        prev_end = e
    if duration - prev_end > 5:
        gaps.append((prev_end, duration))
    if gaps:
        lines.append("Unexplored: " + ", ".join(f"[{s:.0f}-{e:.0f}s]" for s, e in gaps))

    if scratchpad.evidence:
        ev_str = ", ".join(f"@{e['time']:.0f}s(c={e['conf']})" for e in scratchpad.evidence)
        lines.append(f"Evidence: {ev_str}")

    return "\n".join(lines)

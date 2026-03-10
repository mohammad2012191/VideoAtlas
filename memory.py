"""
memory.py — Visual scratchpad, negative memory, and navigation state.
"""

import json
import os
import cv2
import numpy as np
from PIL import Image
from config import INCLUDE_DESC_IN_SCRATCHPAD
import logger as _logger
from logger import log, save_debug_image


# ==========================================
# INDEX <-> LETTER HELPERS
# ==========================================
def _idx_to_letter(idx):
    """Convert 0-based index to letter label: 0->A, 1->B, ..., 25->Z, 26->AA, etc."""
    result = ""
    while True:
        result = chr(ord('A') + idx % 26) + result
        idx = idx // 26 - 1
        if idx < 0:
            break
    return result


def _letter_to_idx(letter):
    """Convert letter label back to 0-based index: A->0, B->1, ..."""
    result = 0
    for ch in letter.upper():
        result = result * 26 + (ord(ch) - ord('A') + 1)
    return result - 1


# ==========================================
# SCRATCHPAD REASONING SIDECAR
# ==========================================
def _save_scratchpad_reasoning(n_items: int, evidence: list, descriptions: list):
    """
    Save a sidecar JSON next to the scratchpad image so visualize_run.py
    can render per-item reasoning in the scratchpad panel.

    The JSON filename mirrors the image filename that save_debug_image() just wrote.
    Because save_debug_image() incremented the counter before writing, the current
    sequence number is _global_counter (already bumped).

    File: <debug_dir>/<seq>_scratchpad_<n>items_reasoning.json
    Schema:
      [
        {
          "letter":      "A",
          "time":        12.3,
          "confidence":  0.9,
          "description": "...",
          "subtitle":    "..."
        },
        ...
      ]
    """
    if _logger._debug_dir is None:
        return
    try:
        seq      = _logger._global_counter          # live value after save_debug_image()
        filename = f"{seq:04d}_scratchpad_{n_items}items_reasoning.json"
        path     = os.path.join(_logger._debug_dir, filename)
        items = []
        for i, e in enumerate(evidence):
            items.append({
                "letter":      _idx_to_letter(i),
                "time":        round(e['time'], 3),
                "confidence":  e['conf'],
                "description": e['desc'],
                "subtitle":    e.get('subtitle', ''),
            })
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        log(f"[SCRATCHPAD] Reasoning saved: {path}")
    except Exception as ex:
        log(f"[SCRATCHPAD] Failed to save reasoning JSON: {ex}")


# ==========================================
# VISUAL SCRATCHPAD
# ==========================================
class VisualScratchpad:
    MIN_TEMPORAL_GAP = 1.0
    MIN_CONFIDENCE   = 0.7
    MAX_EVIDENCE     = 99999

    def __init__(self):
        self.evidence = []

    def add_evidence(self, image, description, confidence, timestamp, subtitle=""):
        if confidence < self.MIN_CONFIDENCE:
            log(f"[SCRATCHPAD] Rejected @{timestamp:.1f}s (conf={confidence} < {self.MIN_CONFIDENCE})")
            return

        for i, existing in enumerate(self.evidence):
            if abs(existing['time'] - timestamp) < self.MIN_TEMPORAL_GAP:
                if confidence > existing['conf']:
                    log(f"[SCRATCHPAD] Replacing @{existing['time']:.1f}s with better @{timestamp:.1f}s")
                    self.evidence[i] = {
                        "image": image, "desc": description,
                        "conf": confidence, "time": timestamp,
                        "subtitle": subtitle
                    }
                    self.evidence.sort(key=lambda e: e['time'])
                    return
                else:
                    log(f"[SCRATCHPAD] Skipped @{timestamp:.1f}s (duplicate of @{existing['time']:.1f}s)")
                    return

        self.evidence.append({
            "image": image, "desc": description,
            "conf": confidence, "time": timestamp,
            "subtitle": subtitle
        })

        if len(self.evidence) > self.MAX_EVIDENCE:
            worst = min(range(len(self.evidence)), key=lambda i: self.evidence[i]['conf'])
            evicted = self.evidence.pop(worst)
            log(f"[SCRATCHPAD] Evicted @{evicted['time']:.1f}s (conf={evicted['conf']}) — over cap")

        self.evidence.sort(key=lambda e: e['time'])
        log(f"[SCRATCHPAD] Added: @{timestamp:.1f}s (Conf: {confidence}) - {description}. "
            f"({len(self.evidence)}/{self.MAX_EVIDENCE})")

    def get_summary(self):
        if not self.evidence:
            return "None yet."
        lines = []
        for e in self.evidence:
            sub_part = f' [Sub: {e["subtitle"][:60]}]' if e.get("subtitle") else ""
            if INCLUDE_DESC_IN_SCRATCHPAD:
                lines.append(f"- @{e['time']:.1f}s: {e['desc'][:80]}...{sub_part} (Conf: {e['conf']})")
            else:
                lines.append(f"- @{e['time']:.1f}s{sub_part} (Conf: {e['conf']})")
        return "\n".join(lines)

    def prune_to_indices(self, keep_indices):
        keep_set = set(keep_indices)
        removed = [e for i, e in enumerate(self.evidence) if i not in keep_set]
        self.evidence = [e for i, e in enumerate(self.evidence) if i in keep_set]
        for e in removed:
            log(f"[SCRATCHPAD] Erased @{e['time']:.1f}s: {e['desc']}...")
        log(f"[SCRATCHPAD] Pruned: kept {len(self.evidence)}, removed {len(removed)}")

    def generate_evidence_grid(self, cell_size=256):
        if not self.evidence:
            placeholder = Image.new('RGB', (cell_size, cell_size), color='gray')
            return placeholder, ["No evidence collected yet."]

        n_items   = len(self.evidence)
        grid_cols = int(np.ceil(np.sqrt(n_items)))
        grid_rows = int(np.ceil(n_items / grid_cols))

        frames       = []
        descriptions = []

        for idx, e in enumerate(self.evidence):
            if isinstance(e['image'], Image.Image):
                img = e['image'].resize((cell_size, cell_size))
            else:
                img = Image.fromarray(e['image']).resize((cell_size, cell_size))

            frame  = np.array(img)
            letter = _idx_to_letter(idx)
            label  = f"[{letter}] @{e['time']:.1f}s"
            font      = cv2.FONT_HERSHEY_SIMPLEX
            scale     = cell_size / 256 * 0.7
            thickness = max(1, int(cell_size / 256 * 2))
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            cv2.rectangle(frame, (0, 0), (tw + 8, th + 12), (0, 0, 0), -1)
            cv2.putText(frame, label, (4, th + 6), font, scale, (255, 255, 255), thickness)

            frames.append(frame)
            sub_part = f' [Sub: {e["subtitle"][:40]}]' if e.get("subtitle") else ""
            if INCLUDE_DESC_IN_SCRATCHPAD:
                descriptions.append(f"[{letter}] @{e['time']:.1f}s: {e['desc']}{sub_part}")
            else:
                descriptions.append(f"[{letter}] @{e['time']:.1f}s{sub_part}")

        while len(frames) < grid_rows * grid_cols:
            frames.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))

        rows    = [np.hstack(frames[r * grid_cols:(r + 1) * grid_cols]) for r in range(grid_rows)]
        pil_img = Image.fromarray(np.vstack(rows))
        save_debug_image(pil_img, f"scratchpad_{n_items}items")

        # Save sidecar JSON with per-item reasoning so visualize_run.py can overlay it
        _save_scratchpad_reasoning(n_items, self.evidence, descriptions)

        return pil_img, descriptions


# ==========================================
# NEGATIVE MEMORY
# ==========================================
class NegativeMemory:
    def __init__(self, total_duration):
        self.total_duration = total_duration
        self.dead_intervals = []

    def add_dead_zone(self, start, end):
        start, end = max(0, start), min(self.total_duration, end)
        self.dead_intervals.append((start, end))
        log(f"[MEMORY] Pruned: {start:.1f}s to {end:.1f}s")

    def is_dead_interval(self, start, end):
        for d_start, d_end in self.dead_intervals:
            overlap = min(end, d_end) - max(start, d_start)
            duration = end - start
            if overlap > 0 and duration > 0 and (overlap / duration) > 0.5:
                return True
        return False

    def coverage_pct(self):
        if not self.dead_intervals:
            return 0.0
        merged = [sorted(self.dead_intervals)[0]]
        for start, end in sorted(self.dead_intervals)[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        covered = sum(e - s for s, e in merged)
        return (covered / self.total_duration) * 100

    def get_summary(self):
        if not self.dead_intervals:
            return "None."
        return ", ".join([f"[{s:.0f}-{e:.0f}s]" for s, e in self.dead_intervals[:10]])


# ==========================================
# NAVIGATION STATE
# ==========================================
class NavigationState:
    def __init__(self, center, span):
        self.center = center
        self.span   = span


class NavigationStack:
    def __init__(self):
        self.stack = []

    def push(self, state):
        self.stack.append(state)
        log(f"[NAV] Pushed: center={state.center:.1f}, span={state.span:.1f} (depth={len(self.stack)})")

    def pop(self):
        if self.stack:
            state = self.stack.pop()
            log(f"[NAV] Popped: center={state.center:.1f}, span={state.span:.1f} (depth={len(self.stack)})")
            return state
        return None

    def depth(self):
        return len(self.stack)

    def get_path(self):
        return " -> ".join([f"[{s.center:.0f}s+/-{s.span/2:.0f}s]" for s in self.stack])
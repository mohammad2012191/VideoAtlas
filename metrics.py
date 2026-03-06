"""
metrics.py — Thread-safe metrics counter for tracking VLM calls, tokens, and frames.
"""

import threading
from config import MASTER_PARAMS_B, WORKER_PARAMS_B


class MetricsCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self.vlm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.master_tokens = 0
        self.worker_tokens = 0
        self.frames_decoded = 0
        self.total_frames = 0
        self.sufficiency_checks = 0
        self.explore_rounds = 0
        self.evidence_count = 0

    def add_call(self, input_tok, output_tok, is_master=False):
        with self._lock:
            self.vlm_calls += 1
            self.input_tokens += input_tok or 0
            self.output_tokens += output_tok or 0
            total = (input_tok or 0) + (output_tok or 0)
            if is_master:
                self.master_tokens += total
            else:
                self.worker_tokens += total

    def add_frames(self, n=1):
        with self._lock:
            self.frames_decoded += n

    def snapshot(self):
        with self._lock:
            total_tokens = self.input_tokens + self.output_tokens
            master_flops = 2 * MASTER_PARAMS_B * 1e9 * self.master_tokens
            worker_flops = 2 * WORKER_PARAMS_B * 1e9 * self.worker_tokens
            estimated_flops = master_flops + worker_flops
            return {
                "vlm_calls":         self.vlm_calls,
                "input_tokens":      self.input_tokens,
                "output_tokens":     self.output_tokens,
                "total_tokens":      total_tokens,
                "master_tokens":     self.master_tokens,
                "worker_tokens":     self.worker_tokens,
                "estimated_tflops":  round(estimated_flops / 1e12, 2),
                "frames_decoded":    self.frames_decoded,
                "total_frames":      self.total_frames,
                "sufficiency_checks": self.sufficiency_checks,
                "explore_rounds":    self.explore_rounds,
                "evidence_count":    self.evidence_count,
            }

    def reset(self):
        with self._lock:
            self.vlm_calls = 0
            self.input_tokens = 0
            self.output_tokens = 0
            self.master_tokens = 0
            self.worker_tokens = 0
            self.frames_decoded = 0
            self.total_frames = 0
            self.sufficiency_checks = 0
            self.explore_rounds = 0
            self.evidence_count = 0


# Global singleton — imported everywhere
metrics = MetricsCounter()

"""
logger.py — Logging utility that writes to both console and a log file,
            and saves debug images to a structured folder.

Every image is prefixed with a zero-padded global sequence number so
filenames sort in true run order by name alone:
    0001_global_grid.jpg
    0002_W1_C0_step0.jpg
    0003_grid_c120s_span240s.jpg
    ...

Call `setup_logger(path)` once at startup, then use `log()` and
`save_debug_image()` anywhere in the codebase.
"""

import os
import threading
from config import DEBUG

_log_file      = None
_debug_dir     = None
_global_counter = 0          # increments with every saved image
_counter_lock  = threading.Lock()


def setup_logger(log_path: str):
    global _log_file, _debug_dir, _global_counter
    _log_file       = open(log_path, "a", encoding="utf-8", buffering=1)
    _global_counter = 0

    base       = os.path.splitext(log_path)[0]
    _debug_dir = base + "_images"
    os.makedirs(_debug_dir, exist_ok=True)

    print(f"[LOGGER] Logging to:     {log_path}")
    print(f"[LOGGER] Debug images -> {_debug_dir}/")

def log(msg: str):
    if DEBUG:
        print(msg)
    if _log_file is not None:
        _log_file.write(msg + "\n")


def save_debug_image(img, name: str):
    """
    Save a PIL Image (or numpy array) with a global sequence prefix.
    e.g.  save_debug_image(img, "W1_C0_step0")  ->  0042_W1_C0_step0.jpg
    Does nothing if setup_logger() has not been called yet.
    """
    if _debug_dir is None:
        return

    from PIL import Image as _Image

    if not isinstance(img, _Image.Image):
        img = _Image.fromarray(img)

    with _counter_lock:
        global _global_counter
        _global_counter += 1
        n = _global_counter

    filename = f"{n:04d}_{name}.jpg"
    path     = os.path.join(_debug_dir, filename)    
    img.save(path, format="JPEG", quality=90)
    log(f"[IMG] Saved: {path}")


def close_logger():
    global _log_file
    if _log_file is not None:
        _log_file.flush()
        _log_file.close()
        _log_file = None
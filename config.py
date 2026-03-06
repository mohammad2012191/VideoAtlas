"""
config.py — All global configuration constants for the Video Explorer system.
Edit these values to control model paths, exploration behavior, and debug output.
"""

import os
import json

# ==========================================
# MODEL CONFIGURATION
# ==========================================
MASTER_MODEL_PATH = "google/gemini-3-flash-preview"
WORKER_MODEL_PATH = "google/gemini-3-flash-preview"

# ==========================================
# VERTEX AI / GCP SETTINGS
# ==========================================
with open("vertex_key.json") as f:
    PROJECT_ID = json.load(f)["project_id"]

LOCATION   = "global"                        # gemini-3-flash-preview requires global
SERVICE_ACCOUNT_FILE = "vertex_key.json"

VERTEX_BASE_URL = (
    f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
    f"/locations/{LOCATION}/endpoints/openapi"
)

# ==========================================
# OUTPUT
# ==========================================
OUTPUT_DIR = "results"       # Directory to save log files and predictions
DEBUG = True                 # True = verbose output to log file and console

# ==========================================
# EXPLORATION SETTINGS
# ==========================================
NUM_GPUS        = 4
GRID_K          = 8          # Grid size: GRID_K x GRID_K cells
EXPLORE_MODE    = "dfs"      # "auto", "bfs", or "dfs"

# DFS settings
NUM_CALLS       = 512        # Total VLM call budget for DFS workers
BUDGET_PER_CELL = 8          # Steps per worker per cell
DFS_MAX_DEPTH   = "auto"     # "auto" stops when cell span < 1s; or set an integer

# BFS settings
BFS_MAX_DEPTH      = "auto"  # "auto" stops when sub-cells < 1s; or set an integer
BFS_MAX_PROMISING  = 2       # Max promising cells a BFS worker can mark
BFS_MAX_EVIDENCE   = 3       # Max evidence items per BFS worker
BFS_BUDGET         = 2       # Steps per BFS worker per cell

# Sufficiency check frequency (1 = every round, 2 = every other round, etc.)
SUFFICIENCY_EVERY_K = 1

# ==========================================
# EXPLORATION PARAMETERS
# ==========================================
PLATEAU          = 10        # Auto-finish after N steps without new evidence
INVESTIGATE_SPAN = 30.0      # Seconds to investigate around an anchor (8x8 grid)
MIN_EXPAND_SPAN  = 1.0       # Don't allow EXPAND below this span (seconds)

# ==========================================
# SCRATCHPAD DISPLAY
# ==========================================
INCLUDE_DESC_IN_SCRATCHPAD = True  # False = evidence grid shows only image + timestamp

# ==========================================
# FLOPS ESTIMATION
# ==========================================
MASTER_PARAMS_B = 3   # Active parameters (billions) for master model
WORKER_PARAMS_B = 3   # Active parameters (billions) for worker model

# ==========================================
# JSON REPAIR RETRIES
# ==========================================
MAX_PROBE_RETRIES = 3

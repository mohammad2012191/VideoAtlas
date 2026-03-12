"""
config.py — All global configuration constants for the Video Explorer system.
Edit these values to control model paths, exploration behavior, and debug output.
"""

import os
import json
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ==========================================
# BACKEND SELECTION
# ==========================================
# Set to "google" to use Google AI API (Gemini API key)
# Set to "vertex" to use Vertex AI (service account JSON)
BACKEND = "vertex"   # "google" or "vertex"

# ==========================================
# GOOGLE AI API SETTINGS (used when BACKEND = "google")
# ==========================================
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# ==========================================
# MODEL CONFIGURATION
# ==========================================
# Vertex AI model paths
VERTEX_MASTER_MODEL_PATH = "google/gemini-3-flash-preview"
VERTEX_WORKER_MODEL_PATH = "google/gemini-3-flash-preview"

# Google AI model names
GOOGLE_MASTER_MODEL_PATH = "models/gemini-3-flash-preview"
GOOGLE_WORKER_MODEL_PATH = "models/gemini-3-flash-preview"

# Resolved at runtime based on BACKEND
MASTER_MODEL_PATH = GOOGLE_MASTER_MODEL_PATH if BACKEND == "google" else VERTEX_MASTER_MODEL_PATH
WORKER_MODEL_PATH = GOOGLE_WORKER_MODEL_PATH if BACKEND == "google" else VERTEX_WORKER_MODEL_PATH

# ==========================================
# VERTEX AI / GCP SETTINGS (used when BACKEND = "vertex")
# ==========================================
SERVICE_ACCOUNT_FILE = "vertex_key.json"

def _load_vertex_config():
    if BACKEND == "vertex":
        if not os.path.isfile(SERVICE_ACCOUNT_FILE):
            raise FileNotFoundError(
                f"Vertex AI backend selected but '{SERVICE_ACCOUNT_FILE}' not found. "
                "Either set BACKEND='google' or provide a valid service account file."
            )
        with open(SERVICE_ACCOUNT_FILE) as f:
            project_id = json.load(f)["project_id"]
        location   = "global"
        base_url   = (
            f"https://aiplatform.googleapis.com/v1/projects/{project_id}"
            f"/locations/{location}/endpoints/openapi"
        )
        return project_id, location, base_url
    return None, None, None

PROJECT_ID, LOCATION, VERTEX_BASE_URL = _load_vertex_config()

# ==========================================
# OUTPUT
# ==========================================
OUTPUT_DIR = "results"       # Directory to save log files and predictions
DEBUG = True                 # True = verbose output to log file and console

# ==========================================
# EXPLORATION SETTINGS
# ==========================================
NUM_WORKERS     = 8          # Number of Agent to use for parallel exploration
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
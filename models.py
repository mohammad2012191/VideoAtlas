"""
models.py — Data classes and enums used across the Video Explorer system.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple


# ==========================================
# CELL STATE
# ==========================================
class CellState(Enum):
    OPEN        = "open"
    IN_PROGRESS = "in_progress"   # Virtual loss: claimed by a worker
    EXPLORED    = "explored"      # Worker finished, with evidence
    DEAD        = "dead"          # No relevant content found


# ==========================================
# FRONTIER CELL
# ==========================================
@dataclass
class FrontierCell:
    cell_id:         int
    start:           float
    end:             float
    probe_score:     float     = 0.0
    state:           CellState = CellState.OPEN
    assigned_worker: int       = -1
    evidence_count:  int       = 0
    reason:          str       = ""


# ==========================================
# SHARED FRONTIER
# ==========================================
class SharedFrontier:
    """
    Manages the frontier of cells to explore.
    Implements virtual loss: claimed cells are IN_PROGRESS so other workers skip them.
    """

    def __init__(self):
        self.cells: dict = {}   # cell_id -> FrontierCell

    def add_cell(self, cell_id, start, end, score=0.0, reason=""):
        self.cells[cell_id] = FrontierCell(
            cell_id=cell_id, start=start, end=end,
            probe_score=score, reason=reason
        )

    def claim(self, cell_id, worker_id):
        if cell_id in self.cells and self.cells[cell_id].state == CellState.OPEN:
            self.cells[cell_id].state           = CellState.IN_PROGRESS
            self.cells[cell_id].assigned_worker = worker_id
            return True
        return False

    def release(self, cell_id, status, evidence_count=0):
        if cell_id in self.cells:
            self.cells[cell_id].state          = status
            self.cells[cell_id].evidence_count = evidence_count

    def get_top_open(self, n):
        open_cells = [c for c in self.cells.values() if c.state == CellState.OPEN]
        open_cells.sort(key=lambda x: x.probe_score, reverse=True)
        return open_cells[:n]

    def has_open(self):
        return any(c.state == CellState.OPEN for c in self.cells.values())

    def get_summary(self):
        counts = {}
        for c in self.cells.values():
            s = c.state.value
            counts[s] = counts.get(s, 0) + 1
        return ", ".join(f"{k}: {v}" for k, v in counts.items())


# ==========================================
# EVIDENCE & WORKER REPORTS
# ==========================================
@dataclass
class EvidenceItem:
    timestamp:   float
    description: str
    confidence:  float
    image:       object   # PIL.Image


@dataclass
class WorkerReport:
    cell_id:         int
    worker_id:       int
    evidence:        List[EvidenceItem]
    dead_zones:      List[Tuple[float, float]]
    status:          CellState
    promising_cells: List = field(default_factory=list)


@dataclass
class BFSItem:
    """An item in the BFS FIFO frontier."""
    start:  float
    end:    float
    reason: str
    depth:  int = 0


@dataclass
class BFSWorkerReport:
    """Report from a BFS worker."""
    worker_id:       int
    bfs_item:        BFSItem
    evidence:        List[EvidenceItem]
    promising_cells: List[BFSItem]

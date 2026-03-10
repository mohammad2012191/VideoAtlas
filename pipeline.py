"""
pipeline.py — Full exploration pipeline for a single question.

Phases:
  1. Master probes the global grid
  2. DFS or BFS exploration with parallel master ranking
  3. Final decision
"""

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    GRID_K, NUM_WORKERS, EXPLORE_MODE,
    NUM_CALLS, BUDGET_PER_CELL, DFS_MAX_DEPTH,
    BFS_MAX_DEPTH, BFS_BUDGET, BFS_MAX_PROMISING,
    MIN_EXPAND_SPAN, SUFFICIENCY_EVERY_K
)
from models import CellState, BFSItem
from memory import VisualScratchpad, NegativeMemory, _letter_to_idx
from navigator import VisualNavigator, blackout_dead_cells, build_progress_text
from workers import probe_cells, master_analyze, worker_explore, worker_explore_bfs
from metrics import metrics
from logger import log, save_debug_image


# ==========================================
# QUERY TASK EXTRACTION
# ==========================================
def extract_search_task(agent, query):
    prompt = f"""You are a planner for a video QA system. Workers scan video frames, subtitles, and text overlays.

QUESTION: "{query}"

Write a short, concrete SEARCH TASK (2-3 sentences max) telling workers exactly what to look for and how to find the answer.

Output ONLY the search task, no prefix:"""

    output = agent._generate([{"role": "user", "content": prompt}], tools=None, max_tokens=256)
    task   = output.strip()
    if task.upper().startswith("SEARCH TASK:"):
        task = task[len("SEARCH TASK:"):].strip()
    if len(task) < 15:
        log("[TASK] Failed to extract task, skipping")
        return query, ""
    log(f"[TASK] Original: {query}...")
    log(f"[TASK] Search task: {task}")
    return query, task


# ==========================================
# MODE SELECTOR
# ==========================================
def select_mode(agent, query):
    prompt = f"""Choose exploration strategy for this video query.

**QUERY:** "{query}"

- **DFS**: Best when the query asks about a specific detail or event in a small part of the video.
- **BFS**: Best when the query requires understanding flow or context across the whole video.

Answer with ONLY "DFS" or "BFS":"""

    output = agent._generate([{"role": "user", "content": prompt}], tools=None, max_tokens=32)
    mode   = "BFS" if "BFS" in output.upper() else "DFS"
    log(f"[MASTER] Mode selected: {mode} (raw: '{output}')")
    return mode


# ==========================================
# MAIN PIPELINE
# ==========================================
def process_question(master, worker_agents, video_path, sub_path, query, candidates=None):
    """
    Run the full exploration pipeline for a single question.
    Returns: (result_dict, mode_used, explore_rounds, coverage_pct)
    """
    original_query, search_task = extract_search_task(master, query)
    if search_task:
        query = f"{original_query}\n\n**SEARCH TASK:** {search_task}"
    else:
        query = original_query

    worker_gpu_ids = sorted(worker_agents.keys())
    num_workers    = len(worker_gpu_ids)

    # ---- Phase 1: Master Analysis ----
    log(f"\n{'='*60}\n[MASTER] Phase 1: Master Analysis (global grid)")
    nav_master     = VisualNavigator(video_path, sub_path, grid_k=GRID_K)
    metrics.total_frames = nav_master.total_frames

    worker_navs    = {gpu_id: VisualNavigator(video_path, sub_path, grid_k=GRID_K)
                      for gpu_id in worker_gpu_ids}

    grid_img, cell_info, _, _ = nav_master.generate_grid_view(
        nav_master.duration / 2, nav_master.duration
    )
    save_debug_image(grid_img, "global_grid")

    cell_span      = nav_master.duration / (GRID_K * GRID_K)
    workers_useful = cell_span >= MIN_EXPAND_SPAN

    if workers_useful:
        ratings         = probe_cells(master, grid_img, cell_info, query, top_n=num_workers)
        master_evidence = []
    else:
        ratings, master_evidence, _ = master_analyze(master, nav_master, grid_img, cell_info, query)

    log(f"[MASTER] Analysis done: {len(ratings)} ratings, {len(master_evidence)} evidence")

    # Build frontier
    from models import SharedFrontier
    frontier  = SharedFrontier()
    score_map = {r['cell_id']: (r.get('score', 0), r.get('reason', '')) for r in ratings}
    for ci in cell_info:
        cid = ci['id']
        score, reason = score_map.get(cid, (0, ''))
        frontier.add_cell(cid, ci['start'], ci['end'], score=score, reason=reason)

    global_scratchpad = VisualScratchpad()
    global_neg_mem    = NegativeMemory(nav_master.duration)
    explored_ranges   = []

    if master_evidence:
        for ev in master_evidence:
            global_scratchpad.add_evidence(ev.image, ev.description, ev.confidence, ev.timestamp)

    # Sufficiency check
    if global_scratchpad.evidence:
        metrics.sufficiency_checks += 1
        if master.check_sufficiency(query, global_scratchpad) == "yes":
            log("[MASTER] Sufficient from global grid — skipping workers.")
            result   = master.final_decide(query, global_scratchpad, candidates=candidates)
            coverage = global_neg_mem.coverage_pct()
            metrics.evidence_count = len(global_scratchpad.evidence)
            return result, "DIRECT", 0, coverage

    # ---- Mode Selection ----
    if not workers_useful:
        mode = "DIRECT"
    elif EXPLORE_MODE == "auto":
        mode = select_mode(master, query)
    else:
        mode = EXPLORE_MODE.upper()

    explore_rounds = 0

    # ============================================================
    # BFS
    # ============================================================
    if mode == "BFS":
        effective_depth = float('inf') if BFS_MAX_DEPTH == "auto" else BFS_MAX_DEPTH
        log(f"\n{'='*60}\n[MASTER] Phase 2: BFS (budget={BFS_BUDGET}/cell)")

        sorted_ratings = sorted(ratings, key=lambda r: r.get('score', 0), reverse=True)
        initial_items  = []
        for r in sorted_ratings:
            cid = r['cell_id']
            ci  = next((c for c in cell_info if c['id'] == cid), None)
            if ci:
                initial_items.append(BFSItem(start=ci['start'], end=ci['end'],
                                             reason=r.get('reason', ''), depth=0))

        batch0_count = min(num_workers, len(initial_items))
        log(f"[BFS] Batch 0: {batch0_count} workers + master scoring in parallel")

        with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
            futures = {}
            for i, item in enumerate(initial_items[:batch0_count]):
                gpu_id = worker_gpu_ids[i]
                futures[executor.submit(
                    worker_explore_bfs, worker_agents[gpu_id], worker_navs[gpu_id],
                    query, item, gpu_id
                )] = ("worker", i)
            futures[executor.submit(
                probe_cells, master, grid_img, cell_info, query, top_n=GRID_K * GRID_K
            )] = ("scorer", -1)

            batch0_reports = []
            full_ratings   = None
            for f in as_completed(futures):
                tt, _ = futures[f]
                try:
                    if tt == "scorer":
                        full_ratings = f.result()
                    else:
                        batch0_reports.append(f.result())
                except Exception as e:
                    log(f"[ERROR] BFS batch 0 {tt} failed: {e}")

        for report in batch0_reports:
            for ev in report.evidence:
                sub_text = nav_master.subs.get_text_for_interval(ev.timestamp - 2, ev.timestamp + 2) or ""
                global_scratchpad.add_evidence(ev.image, ev.description, ev.confidence,
                                               ev.timestamp, subtitle=sub_text)

        already_explored = {(round(i.start, 2), round(i.end, 2)) for i in initial_items[:batch0_count]}
        for item in initial_items[:batch0_count]:
            explored_ranges.append((item.start, item.end))

        bfs_queue = deque()
        if full_ratings:
            for r in sorted(full_ratings, key=lambda x: x.get('score', 0), reverse=True):
                cid = r['cell_id']
                ci  = next((c for c in cell_info if c['id'] == cid), None)
                if ci:
                    key = (round(ci['start'], 2), round(ci['end'], 2))
                    if key not in already_explored:
                        bfs_queue.append(BFSItem(start=ci['start'], end=ci['end'],
                                                 reason=r.get('reason', ''), depth=0))
        for report in batch0_reports:
            for pc in report.promising_cells:
                if pc.end - pc.start >= 1.0 and pc.depth <= effective_depth:
                    bfs_queue.append(pc)

        batch_num = 1
        while bfs_queue:
            batch = [bfs_queue.popleft() for _ in range(min(num_workers, len(bfs_queue)))]
            if not batch:
                break

            log(f"\n{'='*60}\n[BFS BATCH {batch_num}] {len(batch)} cells | Queue: {len(bfs_queue)}")
            current_keys = {(round(i.start, 2), round(i.end, 2)) for i in batch}
            for ci in cell_info:
                if (round(ci['start'], 2), round(ci['end'], 2)) in already_explored | current_keys:
                    ci['status'] = 'DEAD'
            masked_grid = blackout_dead_cells(grid_img, cell_info)
            save_debug_image(masked_grid, f"BFS_batch{batch_num}_masked_grid")
            futures[executor.submit(
                probe_cells, master, masked_grid, cell_info, query, top_n=num_workers
            )] = ("ranker", -1)

            reports       = []
            next_rankings = None
            for f in as_completed(futures):
                tt, _ = futures[f]
                try:
                    if tt == "ranker":
                        next_rankings = f.result()
                    else:
                        reports.append(f.result())
                except Exception as e:
                    log(f"[ERROR] BFS {tt} failed: {e}")

            evidence_before = len(global_scratchpad.evidence)
            for report in reports:
                for ev in report.evidence:
                    sub_text = nav_master.subs.get_text_for_interval(ev.timestamp - 2, ev.timestamp + 2) or ""
                    global_scratchpad.add_evidence(ev.image, ev.description, ev.confidence,
                                                   ev.timestamp, subtitle=sub_text)
                for pc in report.promising_cells:
                    if pc.end - pc.start >= 1.0 and pc.depth <= effective_depth:
                        bfs_queue.append(pc)
            new_evidence = len(global_scratchpad.evidence) > evidence_before

            for item in batch:
                already_explored.add((round(item.start, 2), round(item.end, 2)))
                explored_ranges.append((item.start, item.end))

            # Reorder queue
            if next_rankings and bfs_queue:
                score_map2 = {}
                for r in next_rankings:
                    for ci in cell_info:
                        if ci['id'] == r['cell_id']:
                            score_map2[(round(ci['start'], 2), round(ci['end'], 2))] = r['score']
                            break
                reordered = sorted(bfs_queue, key=lambda i: -score_map2.get(
                    (round(i.start, 2), round(i.end, 2)), 0))
                bfs_queue.clear()
                bfs_queue.extend(reordered)

            if new_evidence and batch_num % SUFFICIENCY_EVERY_K == 0:
                metrics.sufficiency_checks += 1
                progress       = build_progress_text(nav_master, explored_ranges, global_scratchpad)
                masked_grid_ua = blackout_dead_cells(grid_img, cell_info)
                save_debug_image(masked_grid_ua, f"BFS_batch{batch_num}_uncertainty_grid")
                ua = master.uncertainty_analysis(
                    query, global_scratchpad, candidates or [],
                    masked_grid_ua, cell_info, progress, num_workers
                )
                log(f"[MASTER] Uncertainty: {ua['action']} — {ua['reasoning']}")
                if ua['action'] == "FINAL_DECISION":
                    break
                _apply_ua(ua, global_scratchpad, bfs_queue, frontier=None, cell_info=cell_info,
                          already_explored=already_explored, mode="BFS")

            batch_num += 1
        explore_rounds = batch_num

    # ============================================================
    # DFS
    # ============================================================
    elif mode == "DFS":
        dfs_effective_depth = None if DFS_MAX_DEPTH == "auto" else DFS_MAX_DEPTH
        log(f"\n{'='*60}\n[MASTER] Phase 2: DFS (budget={NUM_CALLS}, {BUDGET_PER_CELL}/cell)")

        total_calls_used = 0
        round_idx        = 1

        while frontier.has_open() and total_calls_used < NUM_CALLS:
            remaining   = NUM_CALLS - total_calls_used
            max_w       = min(num_workers, remaining // BUDGET_PER_CELL)
            if max_w <= 0:
                break

            top_cells = frontier.get_top_open(max_w)
            if not top_cells:
                break

            for i, cell in enumerate(top_cells):
                frontier.claim(cell.cell_id, worker_gpu_ids[i % num_workers])

            log(f"\n{'='*60}\n[ROUND {round_idx}] {len(top_cells)} workers | "
                f"Calls: {total_calls_used}/{NUM_CALLS}")

            for ci in cell_info:
                cid = ci['id']
                fc  = frontier.cells.get(cid)
                if fc and fc.state != CellState.OPEN:
                    ci['status'] = 'DEAD'
            masked_grid = blackout_dead_cells(grid_img, cell_info)
            save_debug_image(masked_grid, f"DFS_round{round_idx}_masked_grid")

            with ThreadPoolExecutor(max_workers=len(top_cells) + 1) as executor:
                futures = {}
                for i, cell in enumerate(top_cells):
                    gpu_id = worker_gpu_ids[i % num_workers]
                    futures[executor.submit(
                        worker_explore, worker_agents[gpu_id], worker_navs[gpu_id],
                        query, cell, BUDGET_PER_CELL, gpu_id, max_depth=dfs_effective_depth
                    )] = ("worker", cell.cell_id)
                futures[executor.submit(
                    probe_cells, master, masked_grid, cell_info, query, top_n=num_workers
                )] = ("ranker", -1)

                reports       = []
                next_rankings = None
                for f in as_completed(futures):
                    tt, tid = futures[f]
                    try:
                        if tt == "ranker":
                            next_rankings = f.result()
                        else:
                            reports.append(f.result())
                    except Exception as e:
                        log(f"[ERROR] DFS {tt} {tid} failed: {e}")
                        if tt == "worker":
                            frontier.release(tid, CellState.DEAD)

            total_calls_used += len(top_cells) * BUDGET_PER_CELL
            evidence_before   = len(global_scratchpad.evidence)

            for report in reports:
                frontier.release(report.cell_id, report.status, len(report.evidence))
                for s, e in report.dead_zones:
                    global_neg_mem.add_dead_zone(s, e)
                for ev in report.evidence:
                    sub_text = nav_master.subs.get_text_for_interval(ev.timestamp - 2, ev.timestamp + 2) or ""
                    global_scratchpad.add_evidence(ev.image, ev.description, ev.confidence,
                                                   ev.timestamp, subtitle=sub_text)
                if report.cell_id in frontier.cells:
                    fc = frontier.cells[report.cell_id]
                    explored_ranges.append((fc.start, fc.end))
                for pc in report.promising_cells:
                    if pc.end - pc.start >= MIN_EXPAND_SPAN:
                        for ci in cell_info:
                            if abs(ci['start'] - pc.start) < 1.0 and abs(ci['end'] - pc.end) < 1.0:
                                if ci['id'] in frontier.cells and \
                                   frontier.cells[ci['id']].state == CellState.OPEN:
                                    frontier.cells[ci['id']].probe_score += 2
                                break
            new_evidence = len(global_scratchpad.evidence) > evidence_before

            if next_rankings:
                for r in next_rankings:
                    cid = r['cell_id']
                    if cid in frontier.cells and frontier.cells[cid].state == CellState.OPEN:
                        frontier.cells[cid].probe_score = r['score']

            log(f"[DFS] Evidence: {len(global_scratchpad.evidence)} | "
                f"Frontier: {frontier.get_summary()}")

            if new_evidence and round_idx % SUFFICIENCY_EVERY_K == 0:
                metrics.sufficiency_checks += 1
                progress       = build_progress_text(nav_master, explored_ranges, global_scratchpad)
                masked_grid_ua = blackout_dead_cells(grid_img, cell_info)
                save_debug_image(masked_grid_ua, f"DFS_round{round_idx}_uncertainty_grid")
                ua = master.uncertainty_analysis(
                    query, global_scratchpad, candidates or [],
                    masked_grid_ua, cell_info, progress, num_workers
                )
                log(f"[MASTER] Uncertainty: {ua['action']} — {ua['reasoning']}")
                if ua['action'] == "FINAL_DECISION":
                    break
                _apply_ua(ua, global_scratchpad, bfs_queue=None, frontier=frontier,
                          cell_info=cell_info, already_explored=None, mode="DFS")

            round_idx += 1

        explore_rounds = round_idx
        log(f"[DFS] Finished: {total_calls_used}/{NUM_CALLS} calls, {round_idx} rounds")

    # ---- Phase 3: Final Decision ----
    log(f"\n{'='*60}\n[MASTER] Phase 3: Final Decision (mode={mode})")
    result   = master.final_decide(query, global_scratchpad, candidates=candidates)
    coverage = global_neg_mem.coverage_pct()
    metrics.evidence_count  = len(global_scratchpad.evidence)
    metrics.explore_rounds  = explore_rounds

    log(f"\nFINAL SCRATCHPAD ({len(global_scratchpad.evidence)} items):")
    log(global_scratchpad.get_summary())
    log(f"COVERAGE: {coverage:.1f}%")
    log(f"[ANSWER] {result.get('answer')}")
    log(f"[REASONING] {result.get('reasoning')}")

    return result, mode, explore_rounds, coverage


# ==========================================
# UNCERTAINTY ANALYSIS APPLY HELPER
# ==========================================
def _apply_ua(ua, global_scratchpad, bfs_queue, frontier, cell_info, already_explored, mode):
    """Apply erase and explore suggestions from uncertainty analysis."""
    if ua.get('erase'):
        erase_indices = set()
        for item in ua['erase']:
            if isinstance(item, str):
                erase_indices.add(_letter_to_idx(item))
            elif isinstance(item, int):
                erase_indices.add(item)
        keep = [i for i in range(len(global_scratchpad.evidence)) if i not in erase_indices]
        global_scratchpad.prune_to_indices(keep)

    for suggestion in ua.get('explore', []):
        if mode == "BFS" and bfs_queue is not None:
            if isinstance(suggestion, int):
                ci = next((c for c in cell_info if c['id'] == suggestion), None)
                if ci:
                    key = (round(ci['start'], 2), round(ci['end'], 2))
                    if key not in already_explored:
                        bfs_queue.appendleft(BFSItem(start=ci['start'], end=ci['end'],
                                                     reason="master suggestion", depth=0))
            elif isinstance(suggestion, dict) and 'start' in suggestion:
                bfs_queue.appendleft(BFSItem(start=suggestion['start'], end=suggestion['end'],
                                             reason="master exploitation target", depth=0))
        elif mode == "DFS" and frontier is not None:
            if isinstance(suggestion, int):
                if suggestion in frontier.cells and frontier.cells[suggestion].state == CellState.OPEN:
                    frontier.cells[suggestion].probe_score += 5
            elif isinstance(suggestion, dict) and 'start' in suggestion:
                tmp_id = 1000 + len(frontier.cells)
                frontier.add_cell(tmp_id, suggestion['start'], suggestion['end'],
                                  score=10, reason="master exploitation target")

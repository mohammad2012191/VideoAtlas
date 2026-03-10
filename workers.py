"""
workers.py — Worker exploration functions and master probe/analyze helpers.

Covers:
  - probe_cells(): master rates all grid cells by relevance
  - master_analyze(): master uses tools on the global grid
  - worker_explore(): DFS worker explores an assigned cell
  - worker_explore_bfs(): BFS worker scans a region
"""

import random
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    GRID_K, BUDGET_PER_CELL, PLATEAU, INVESTIGATE_SPAN,
    BFS_BUDGET, BFS_MAX_PROMISING, BFS_MAX_EVIDENCE,
    MIN_EXPAND_SPAN, MAX_PROBE_RETRIES, NUM_WORKERS
)
from models import CellState, EvidenceItem, WorkerReport, BFSItem, BFSWorkerReport
from memory import NavigationState, NavigationStack
from navigator import build_context_str
from tools import (
    TOOL_EXPAND, TOOL_BACKTRACK, TOOL_ZOOM, TOOL_INVESTIGATE,
    TOOL_ADD_TO_SCRATCHPAD, TOOL_MARK_PROMISING, TOOL_FINISHED,
    get_master_tools, get_exploration_tools
)
from logger import log, save_debug_image


# ==========================================
# PROBE CELLS
# ==========================================
def _try_parse_ratings(agent, output):
    result  = agent._parse_json(output)
    entries = result.get("top", result.get("ratings", []))
    ratings = []
    for rank, e in enumerate(entries):
        ratings.append({
            "cell_id": e.get("id", e.get("cell_id", 0)),
            "score":   len(entries) - rank,
            "reason":  e.get("r", e.get("reason", ""))
        })
    if not ratings:
        raise ValueError("Empty ratings")
    return ratings


def probe_cells(agent, grid_img, cell_info, query, top_n=None):
    if top_n is None:
        top_n = NUM_WORKERS
    context_str = build_context_str(cell_info)

    prompt = f"""You are analyzing a {GRID_K}x{GRID_K} grid of frames from a SINGLE video (chronological left-to-right, top-to-bottom).

**QUERY:** "{query}"

**GRID CELLS:**
{context_str}

List ONLY the top {top_n} cells most likely to help complete the SEARCH TASK.
For each cell briefly explain WHY it is promising, then give its ID.

**OUTPUT (raw JSON — reason FIRST, then id):**
{{"top": [{{"r": "<one sentence: why promising>", "id": <cell_id>}}, ...]}}"""

    messages = [{"role": "user", "content": [
        {"type": "image", "image": grid_img},
        {"type": "text",  "text":  prompt}
    ]}]

    output = agent._generate(messages, tools=None, max_tokens=2048)
    log(f"[PROBE] Raw: {output}")
    try:
        return _try_parse_ratings(agent, output)
    except Exception as e:
        log(f"[PROBE] Parse failed: {e}")

    for attempt in range(MAX_PROBE_RETRIES):
        fix_prompt = f"""Fix this malformed JSON and return ONLY valid JSON.

Expected: {{"top": [{{"r": "<reason>", "id": <int>}}, ...]}}

Broken:
{output[:3000]}

**OUTPUT (valid JSON only):**"""
        fixed = agent._generate([{"role": "user", "content": fix_prompt}], tools=None, max_tokens=2048)
        try:
            return _try_parse_ratings(agent, fixed)
        except Exception as e:
            log(f"[PROBE] Repair attempt {attempt+1}/{MAX_PROBE_RETRIES} failed: {e}")
            output = fixed

    log("[PROBE] All retries failed. Using random selection.")
    shuffled = list(range(len(cell_info)))
    random.shuffle(shuffled)
    return [{"cell_id": i, "score": 1, "reason": "random"} for i in shuffled[:top_n]]


# ==========================================
# MASTER ANALYZE (with tools, global grid)
# ==========================================
def master_analyze(agent, nav, grid_img, cell_info, query):
    context_str  = build_context_str(cell_info)
    master_tools = get_master_tools()

    prompt = f"""You are the MASTER analyzing a {GRID_K}x{GRID_K} grid covering an entire {nav.duration:.0f}s video (chronological left-to-right, top-to-bottom).

**QUERY:** "{query}"

**GRID:**
{context_str}

If you can already find what the search task asks for:
- Use ADD_TO_SCRATCHPAD to save evidence.
- Description format: '<what you observe>. This helps because <reason>.'
- Always end with FINISHED."""

    messages       = [{"role": "user", "content": [
        {"type": "image", "image": grid_img},
        {"type": "text",  "text":  prompt}
    ]}]

    evidence            = []
    master_found_evidence = False

    for step in range(3):
        try:
            output    = agent._generate(messages, tools=master_tools, max_tokens=2048)
            log(f"[MASTER] Step {step} RAW: {output}...")
            decisions = agent._parse_tool_calls(output)
            log(f"[MASTER] Step {step} PARSED: {[d.get('action') for d in decisions]}")
        except Exception as e:
            log(f"[MASTER] Step {step} parse error: {e}")
            fix_msgs = [{"role": "user", "content":
                f"Fix this malformed tool call JSON:\n{output}\nReturn ONLY corrected JSON:"}]
            try:
                fixed_output = agent._generate(fix_msgs, tools=None, max_tokens=2048)
                decisions    = agent._parse_tool_calls(fixed_output)
            except Exception as e2:
                log(f"[MASTER] Repair also failed: {e2}")
                break

        messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content": "Continue. Use ADD_TO_SCRATCHPAD or FINISHED."})

        finished = False
        for decision in decisions:
            action = decision.get('action')
            if action == "ADD_TO_SCRATCHPAD":
                for item in decision.get('items', []):
                    ts        = item.get('timestamp', nav.duration / 2)
                    desc      = item.get('description', '')
                    conf      = item.get('confidence', 0.75)
                    frame_img = Image.fromarray(nav.get_frame(ts))
                    evidence.append(EvidenceItem(ts, desc, conf, frame_img))
                    log(f"[MASTER] EVIDENCE: @{ts:.1f}s conf={conf} '{desc}'")
                master_found_evidence = True
            elif action == "FINISHED":
                log(f"[MASTER] FINISHED (evidence={len(evidence)})")
                finished = True
                break
        if finished:
            break

    log("[MASTER] Getting probe ratings for frontier...")
    ratings = probe_cells(agent, grid_img, cell_info, query)
    return ratings, evidence, master_found_evidence


# ==========================================
# DFS WORKER
# ==========================================
def worker_explore(agent, nav, query, cell, budget, worker_id, max_depth=None):
    nav_stack = NavigationStack()
    center    = (cell.start + cell.end) / 2
    span      = cell.end - cell.start

    evidence        = []
    dead_zones      = []
    promising_cells = []
    no_evidence_steps = 0
    prev_summary    = ""

    tag = f"[W{worker_id}|C{cell.cell_id}]"
    log(f"  {tag} Start [{cell.start:.1f}-{cell.end:.1f}s] budget={budget}")

    for step in range(budget):
        grid_img, cell_info, _, _ = nav.generate_grid_view(center, span)
        save_debug_image(grid_img, f"W{worker_id}_C{cell.cell_id}_step{step}")
        context_str  = build_context_str(cell_info)
        depth        = nav_stack.depth()
        step_tools   = get_exploration_tools(span, depth, max_depth=max_depth)
        available    = [t['function']['name'] for t in step_tools]

        pct_start = cell.start / nav.duration * 100
        pct_end   = cell.end   / nav.duration * 100
        history   = f"\n**Previous step:** {prev_summary}" if prev_summary else ""

        user_prompt = f"""You are exploring a region of a SINGLE {nav.duration:.0f}s video.

**QUERY:** "{query}"
**POSITION:** {cell.start:.1f}s - {cell.end:.1f}s (≈{pct_start:.0f}%-{pct_end:.0f}%)
**Depth:** {depth} | **Span:** {span:.1f}s | **Steps remaining:** {budget - step}{history}

**GRID ({GRID_K}x{GRID_K}, chronological left-to-right, top-to-bottom):**
{context_str}

**AVAILABLE TOOLS:** {', '.join(available)}

ADD_TO_SCRATCHPAD rules:
- Only save evidence that directly helps complete the search task.
- Description: '<what you observe>. This helps because <reason>.'
- Confidence: 0.5=tangential, 0.8=directly addresses task, 1.0=definitive.

Select the best action."""

        messages = [{"role": "user", "content": [
            {"type": "image", "image": grid_img},
            {"type": "text",  "text":  user_prompt}
        ]}]

        try:
            output    = agent._generate(messages, tools=step_tools, max_tokens=2048)
            log(f"  {tag} Step {step} RAW: {output}...")
            decisions = agent._parse_tool_calls(output)
            log(f"  {tag} Step {step} PARSED: {[d.get('action') for d in decisions]}")
        except Exception as e:
            log(f"  {tag} Step {step} PARSE ERROR: {e}")
            fix_msgs = [{"role": "user", "content":
                f"Fix this malformed tool call:\n{output}\nReturn ONLY corrected JSON:"}]
            try:
                fixed    = agent._generate(fix_msgs, tools=None, max_tokens=2048)
                decisions = agent._parse_tool_calls(fixed)
            except Exception as e2:
                log(f"  {tag} Repair failed: {e2}")
                decisions = [{"action": "FINISHED"}]

        evidence_this_step  = False
        finished            = False
        investigate_used    = False
        step_actions        = []

        for decision in decisions:
            action = decision.get('action')

            if action == "EXPAND" and "EXPAND" in available:
                cid = decision.get('cell_id', -1)
                if 0 <= cid < len(cell_info):
                    ci     = cell_info[cid]
                    nav_stack.push(NavigationState(center, span))
                    center = (ci['start'] + ci['end']) / 2
                    span   = (ci['end'] - ci['start']) * GRID_K
                    step_actions.append(f"Expanded into cell {cid} [{ci['start']:.0f}-{ci['end']:.0f}s]")
                break

            elif action == "BACKTRACK" and "BACKTRACK" in available:
                prev = nav_stack.pop()
                if prev:
                    center = prev.center
                    span   = prev.span
                    step_actions.append(f"Backtracked to {center:.0f}s")
                break

            elif action == "ZOOM" and "ZOOM" in available:
                ts   = decision.get('timestamp', center)
                dur  = decision.get('duration', 0.0)
                log(f"  {tag} ZOOM @ {ts:.1f}s")
                full_img, sub_text, _, _ = nav.get_full_frame(ts, dur)

                zoom_prompt = f"""ZOOM at {ts:.1f}s. Analyze this frame for the search task.

**QUERY:** "{query}"
**Subtitles:** {sub_text}

If relevant, use ADD_TO_SCRATCHPAD. Otherwise use FINISHED.
Description: '<what you observe>. This helps because <reason>.'"""

                zoom_msgs  = [{"role": "user", "content": [
                    {"type": "image", "image": full_img},
                    {"type": "text",  "text":  zoom_prompt}
                ]}]
                zoom_tools = [TOOL_ADD_TO_SCRATCHPAD, TOOL_FINISHED]
                try:
                    zoom_out  = agent._generate(zoom_msgs, tools=zoom_tools, max_tokens=2048)
                    zoom_decs = agent._parse_tool_calls(zoom_out)
                    for zd in zoom_decs:
                        if zd.get('action') == 'ADD_TO_SCRATCHPAD':
                            for item in zd.get('items', []):
                                t = item.get('timestamp', ts)
                                d = item.get('description', '')
                                c = item.get('confidence', 0.75)
                                evidence.append(EvidenceItem(t, d, c, full_img))
                                evidence_this_step = True
                            step_actions.append(f"Zoomed @{ts:.0f}s, found evidence")
                        elif zd.get('action') == 'FINISHED':
                            finished = True
                except Exception as e:
                    log(f"  {tag} Zoom parse error: {e}")

            elif action == "INVESTIGATE" and "INVESTIGATE" in available and not investigate_used:
                investigate_used = True
                anchor_ts  = decision.get('timestamp', center)
                direction  = decision.get('direction', 'after')
                reason     = decision.get('reason', '')

                inv_center = (
                    min(anchor_ts + INVESTIGATE_SPAN / 2, nav.duration - 1)
                    if direction == "after"
                    else max(anchor_ts - INVESTIGATE_SPAN / 2, 1)
                )
                log(f"  {tag} INVESTIGATE {direction} @{anchor_ts:.1f}s: {reason}")
                inv_grid, inv_info, _, _ = nav.generate_grid_view(inv_center, INVESTIGATE_SPAN)
                inv_context = build_context_str(inv_info)

                inv_prompt = f"""INVESTIGATE: Looking {direction.upper()} @{anchor_ts:.1f}s for the answer.

**Reason:** {reason}
**QUERY:** "{query}"

**GRID:**
{inv_context}

State clearly what you see and which candidate it points to.
Description: '<exact scene>. This means the answer is X because <direct reasoning>.'
Confidence: 0.7=uncertain, 0.9=strong, 1.0=definitive.
If nothing answers the question, call FINISHED."""

                inv_msgs  = [{"role": "user", "content": [
                    {"type": "image", "image": inv_grid},
                    {"type": "text",  "text":  inv_prompt}
                ]}]
                inv_tools = [TOOL_ADD_TO_SCRATCHPAD, TOOL_FINISHED]
                inv_out   = agent._generate(inv_msgs, tools=inv_tools, max_tokens=2048)
                inv_decs  = None
                try:
                    inv_decs = agent._parse_tool_calls(inv_out)
                except Exception as e:
                    log(f"  {tag} INVESTIGATE parse failed: {e}")

                if not inv_decs:
                    retry_msgs = inv_msgs + [
                        {"role": "assistant", "content": inv_out},
                        {"role": "user", "content":
                            "Call ADD_TO_SCRATCHPAD with what you found, or FINISHED. No plain text."}
                    ]
                    try:
                        inv_decs = agent._parse_tool_calls(
                            agent._generate(retry_msgs, tools=inv_tools, max_tokens=2048))
                    except Exception as e2:
                        log(f"  {tag} INVESTIGATE retry also failed: {e2}")

                if inv_decs:
                    for inv_d in inv_decs:
                        if inv_d.get('action') == 'ADD_TO_SCRATCHPAD':
                            for item in inv_d.get('items', []):
                                t = item.get('timestamp', anchor_ts)
                                d = item.get('description', '')
                                c = item.get('confidence', 0.75)
                                frame_img = Image.fromarray(nav.get_frame(t))
                                evidence.append(EvidenceItem(t, d, c, frame_img))
                                evidence_this_step = True
                            step_actions.append(f"Investigated {direction} @{anchor_ts:.0f}s")

            elif action == "ADD_TO_SCRATCHPAD":
                for item in decision.get('items', []):
                    ts   = item.get('timestamp', center)
                    desc = item.get('description', '')
                    conf = item.get('confidence', 0.75)
                    frame_img = Image.fromarray(nav.get_frame(ts))
                    evidence.append(EvidenceItem(ts, desc, conf, frame_img))
                    evidence_this_step = True
                    log(f"  {tag} EVIDENCE: @{ts:.1f}s conf={conf} '{desc}'")
                step_actions.append(f"Saved {len(decision.get('items', []))} evidence items")

            elif action == "MARK_PROMISING":
                cell_ids = decision.get('cell_ids', decision.get('values', []))[:BFS_MAX_PROMISING]
                for cid in cell_ids:
                    if 0 <= cid < len(cell_info):
                        ci = cell_info[cid]
                        promising_cells.append(BFSItem(
                            start=ci['start'], end=ci['end'],
                            reason=f"Marked by W{worker_id} at depth {depth}",
                            depth=depth + 1
                        ))
                step_actions.append(f"Marked {len(cell_ids)} promising cells")

            elif action == "FINISHED":
                log(f"  {tag} FINISHED (self-declared)")
                finished = True
                break

        prev_summary = "; ".join(step_actions) if step_actions else "No relevant content found."

        if finished:
            break
        no_evidence_steps = 0 if evidence_this_step else no_evidence_steps + 1
        if no_evidence_steps >= PLATEAU:
            log(f"  {tag} PLATEAU -> auto-finish")
            break

    status = CellState.EXPLORED if evidence else CellState.DEAD
    if status == CellState.DEAD:
        dead_zones.append((cell.start, cell.end))

    log(f"  {tag} Done: {len(evidence)} evidence, {len(promising_cells)} promising, "
        f"status={status.value}")
    return WorkerReport(cell.cell_id, worker_id, evidence, dead_zones, status, promising_cells)


# ==========================================
# BFS WORKER
# ==========================================
def worker_explore_bfs(agent, nav, query, bfs_item, worker_id):
    center    = (bfs_item.start + bfs_item.end) / 2
    span      = bfs_item.end - bfs_item.start
    tag       = f"[BFS-W{worker_id}]"

    log(f"  {tag} Scanning [{bfs_item.start:.1f}-{bfs_item.end:.1f}s] "
        f"budget={BFS_BUDGET} (depth={bfs_item.depth})")

    evidence        = []
    promising_cells = []
    nav_stack       = NavigationStack()
    prev_summary    = ""

    for step in range(BFS_BUDGET):
        grid_img, cell_info, _, _ = nav.generate_grid_view(center, span)
        save_debug_image(grid_img, f"BFSW{worker_id}_depth{bfs_item.depth}_step{step}")
        context_str = build_context_str(cell_info)
        depth       = nav_stack.depth()
        bfs_tools   = get_exploration_tools(span, depth, max_depth=None)
        available   = [t['function']['name'] for t in bfs_tools]

        pct_start = bfs_item.start / nav.duration * 100
        pct_end   = bfs_item.end   / nav.duration * 100
        history   = f"\n**Previous step:** {prev_summary}" if prev_summary else ""

        user_prompt = f"""You are scanning a region of a SINGLE {nav.duration:.0f}s video.

**QUERY:** "{query}"
**POSITION:** {bfs_item.start:.1f}s - {bfs_item.end:.1f}s (≈{pct_start:.0f}%-{pct_end:.0f}%)
**Depth:** {depth} | **Span:** {span:.1f}s | **Steps remaining:** {BFS_BUDGET - step}{history}

**GRID ({GRID_K}x{GRID_K}, chronological):**
{context_str}

**AVAILABLE TOOLS:** {', '.join(available)}

Only save evidence that directly helps complete the search task.
Use MARK_PROMISING only for cells likely to contain the answer.

Select the best action."""

        messages = [{"role": "user", "content": [
            {"type": "image", "image": grid_img},
            {"type": "text",  "text":  user_prompt}
        ]}]

        try:
            output    = agent._generate(messages, tools=bfs_tools, max_tokens=2048)
            log(f"  {tag} Step {step} RAW: {output}...")
            decisions = agent._parse_tool_calls(output)
            log(f"  {tag} Step {step} PARSED: {[d.get('action') for d in decisions]}")
        except Exception as e:
            log(f"  {tag} Step {step} PARSE ERROR: {e}")
            fix_msgs = [{"role": "user", "content":
                f"Fix this malformed tool call:\n{output}\nReturn ONLY corrected JSON:"}]
            try:
                fixed    = agent._generate(fix_msgs, tools=None, max_tokens=256)
                decisions = agent._parse_tool_calls(fixed)
            except:
                decisions = [{"action": "FINISHED"}]

        finished     = False
        step_actions = []

        for decision in decisions:
            action = decision.get('action')

            if action == "EXPAND" and "EXPAND" in available:
                cid = decision.get('cell_id', -1)
                if 0 <= cid < len(cell_info):
                    ci     = cell_info[cid]
                    nav_stack.push(NavigationState(center, span))
                    center = (ci['start'] + ci['end']) / 2
                    span   = (ci['end'] - ci['start']) * GRID_K
                    step_actions.append(f"Expanded into cell {cid}")
                break

            elif action == "BACKTRACK" and "BACKTRACK" in available:
                prev = nav_stack.pop()
                if prev:
                    center = prev.center
                    span   = prev.span
                    step_actions.append(f"Backtracked to {center:.0f}s")
                break

            elif action == "ADD_TO_SCRATCHPAD":
                for item in decision.get('items', [])[:BFS_MAX_EVIDENCE]:
                    ts   = item.get('timestamp', center)
                    desc = item.get('description', '')
                    conf = item.get('confidence', 0.75)
                    frame_img = Image.fromarray(nav.get_frame(ts))
                    evidence.append(EvidenceItem(ts, desc, conf, frame_img))
                    log(f"  {tag} EVIDENCE: @{ts:.1f}s conf={conf} '{desc}'")
                step_actions.append(f"Saved evidence")

            elif action == "MARK_PROMISING":
                cell_ids = decision.get('cell_ids', decision.get('values', []))[:BFS_MAX_PROMISING]
                for cid in cell_ids:
                    if 0 <= cid < len(cell_info):
                        ci = cell_info[cid]
                        promising_cells.append(BFSItem(
                            start=ci['start'], end=ci['end'],
                            reason=f"Marked by BFS-W{worker_id} (depth {bfs_item.depth})",
                            depth=bfs_item.depth + 1
                        ))
                step_actions.append(f"Marked promising")

            elif action == "FINISHED":
                log(f"  {tag} FINISHED (evidence={len(evidence)}, promising={len(promising_cells)})")
                finished = True
                break

        prev_summary = "; ".join(step_actions) if step_actions else "No relevant content found."
        if finished:
            break

    return BFSWorkerReport(worker_id, bfs_item, evidence, promising_cells)

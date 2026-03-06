"""
main.py — Entry point for single-video question answering.

Usage:
    python main.py

You will be prompted to enter:
  - Path to the video file
  - Path to the subtitle JSON file (or leave empty for none)
  - Your question
  - Answer options (comma-separated)

All debug output is printed to the console AND written to a timestamped log file in the results/ folder.
"""

import os
import sys
import json
import time

from config import NUM_GPUS, OUTPUT_DIR
from logger import setup_logger, log, close_logger
from metrics import metrics
from agents import MasterAgent, WorkerAgent
from pipeline import process_question
from visualize_run import build_video


# ==========================================
# RESULT HELPERS
# ==========================================
def extract_choice(result, candidates):
    choice = result.get("choice", -1)
    answer = result.get("answer", "")

    if isinstance(choice, int) and 0 <= choice < len(candidates):
        return choice, candidates[choice]
    if isinstance(choice, str):
        try:
            ci = int(choice)
            if 0 <= ci < len(candidates):
                return ci, candidates[ci]
        except ValueError:
            pass

    answer_lower = answer.lower().strip()
    for i, c in enumerate(candidates):
        if c.lower().strip() == answer_lower:
            return i, c
    for i, c in enumerate(candidates):
        if c.lower().strip() in answer_lower or answer_lower in c.lower().strip():
            return i, c

    return -1, answer


def print_metrics(m):
    log("\n" + "="*60)
    log("METRICS SUMMARY")
    log("="*60)
    for k, v in m.items():
        log(f"  {k:<25}: {v}")
    log("="*60)


# ==========================================
# USER INPUT COLLECTION
# ==========================================
def collect_inputs():
    print("\n" + "="*60)
    print("  VIDEO EXPLORER — Single Video Q&A")
    print("="*60)

    # Video path
    while True:
        video_path = input("\nVideo file path: ").strip()
        if os.path.isfile(video_path):
            break
        print(f"  [!] File not found: {video_path}")

    # Subtitle path (optional)
    sub_path = input("Subtitle JSON path (leave empty to skip): ").strip()
    if sub_path and not os.path.isfile(sub_path):
        print(f"  [!] Subtitle file not found, skipping: {sub_path}")
        sub_path = ""

    # Question
    while True:
        question = input("\nQuestion: ").strip()
        if question:
            break
        print("  [!] Question cannot be empty.")

    # Candidates
    print("\nEnter answer options, one per line.")
    print("Press Enter twice when done (minimum 2 options).")
    candidates = []
    while True:
        opt = input(f"  Option {len(candidates)}: ").strip()
        if opt == "":
            if len(candidates) >= 2:
                break
            print("  [!] Please enter at least 2 options.")
        else:
            candidates.append(opt)

    return video_path, sub_path, question, candidates


# ==========================================
# MAIN
# ==========================================
def main():
    # Collect inputs
    video_path, sub_path, question, candidates = collect_inputs()

    # Setup output directory and log file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp  = time.strftime("%Y%m%d_%H%M%S")
    log_path   = os.path.join(OUTPUT_DIR, f"run_{timestamp}.log")
    result_path = os.path.join(OUTPUT_DIR, f"result_{timestamp}.json")
    setup_logger(log_path)

    log("="*60)
    log("VIDEO EXPLORER — Run started")
    log(f"  Video:      {video_path}")
    log(f"  Subtitles:  {sub_path or '(none)'}")
    log(f"  Question:   {question}")
    log(f"  Candidates: {candidates}")
    log(f"  Log file:   {log_path}")
    log("="*60)

    # Build query string
    candidates_str = ", ".join([f"{i}: '{c}'" for i, c in enumerate(candidates)])
    query = f"In the given video, {question} Choose a single answer: {candidates_str}"

    # Instantiate agents
    print(f"\nLoading agents (NUM_GPUS={NUM_GPUS})...")
    t0           = time.time()
    master_agent = MasterAgent(gpu_id=0)
    worker_agents = {}
    for gpu_id in range(1, NUM_GPUS):
        worker_agents[gpu_id] = WorkerAgent(gpu_id=gpu_id)
    print(f"Agents ready in {time.time() - t0:.1f}s\n")

    # Reset metrics and run
    metrics.reset()
    t_start = time.time()

    try:
        result, mode_used, explore_rounds, coverage = process_question(
            master_agent, worker_agents, video_path, sub_path, query, candidates=candidates
        )

        pred_choice, pred_answer = extract_choice(result, candidates)
        wall_time = time.time() - t_start

        # Build output record
        m = metrics.snapshot()
        m["wall_time_s"]   = round(wall_time, 2)
        m["coverage_pct"]  = round(coverage, 2)
        m["mode_used"]     = mode_used

        output_record = {
            "question":         question,
            "candidates":       candidates,
            "predicted_choice": pred_choice,
            "predicted_answer": pred_answer,
            "reasoning":        result.get("reasoning", ""),
            "metrics":          m,
        }

        # Final summary to log
        log("\n" + "="*60)
        log("FINAL RESULT")
        log("="*60)
        log(f"  Question:         {question}")
        for i, c in enumerate(candidates):
            marker = " ← PREDICTED" if i == pred_choice else ""
            log(f"  Option {i}: {c}{marker}")
        log(f"\n  Predicted answer: {pred_answer} (index {pred_choice})")
        log(f"  Reasoning:        {result.get('reasoning', '')[:500]}")
        print_metrics(m)

        # Save result JSON
        with open(result_path, "w") as f:
            json.dump(output_record, f, indent=2)
        log(f"\nResult saved to: {result_path}")
        log(f"Full log saved to: {log_path}")

        # Also print final result to console clearly
        print("\n" + "="*60)
        print("ANSWER")
        print("="*60)
        print(f"  Predicted: [{pred_choice}] {pred_answer}")
        print(f"  Log:       {log_path}")
        print(f"  Result:    {result_path}")
        print("="*60)

        # Auto-generate replay video
        images_dir = os.path.splitext(log_path)[0] + "_images"
        if os.path.isdir(images_dir):
            print("\n" + "="*60)
            print("GENERATING REPLAY VIDEO")
            print("="*60)
            try:
                replay_path = build_video(
                    run_folder=images_dir,
                    output_path=None,       # saves to <images_dir>/replay.mp4
                    fps=1.5,
                    result_json_path=result_path,
                )
                if replay_path:
                    print(f"  Replay:    {replay_path}")
                    log(f"Replay video saved to: {replay_path}")
            except Exception as ve:
                print(f"  [!] Replay video generation failed: {ve}")
                log(f"[WARN] Replay video generation failed: {ve}")
        else:
            log(f"[INFO] No debug images folder at {images_dir} — skipping replay video.")

    except Exception as e:
        wall_time = time.time() - t_start
        log(f"\n[ERROR] Pipeline failed after {wall_time:.1f}s: {e}")
        import traceback
        log(traceback.format_exc())
        print(f"\n[ERROR] {e}")
        print(f"See log: {log_path}")
        sys.exit(1)
    finally:
        close_logger()


if __name__ == "__main__":
    main()
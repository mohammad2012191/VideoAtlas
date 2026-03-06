"""
extract_subtitles.py — Extract embedded subtitles from a video file and save as JSON.

The output JSON format is directly compatible with the Video Explorer pipeline
(SubtitleReader in navigator.py).

Usage:
    python extract_subtitles.py
    python extract_subtitles.py --video myvideo.mp4
    python extract_subtitles.py --video myvideo.mp4 --output subs.json
    python extract_subtitles.py --video myvideo.mp4 --track 1   # pick a specific subtitle track

Requires: ffmpeg installed and on PATH.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile


# ==========================================
# HELPERS
# ==========================================

def run(cmd, capture=True):
    """Run a shell command. Returns (stdout, stderr, returncode)."""
    result = subprocess.run(
        cmd, shell=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True
    )
    return result.stdout or "", result.stderr or "", result.returncode


def list_subtitle_tracks(video_path):
    """
    Use ffprobe to list all subtitle streams in the video.
    Returns a list of dicts: [{index, codec, language, title}, ...]
    """
    cmd = (
        f'ffprobe -v error -select_streams s '
        f'-show_entries stream=index,codec_name:stream_tags=language,title '
        f'-of json "{video_path}"'
    )
    stdout, stderr, rc = run(cmd)
    if rc != 0 or not stdout.strip():
        return []

    try:
        data    = json.loads(stdout)
        streams = data.get("streams", [])
        tracks  = []
        for i, s in enumerate(streams):
            tags = s.get("tags", {})
            tracks.append({
                "stream_index": s.get("index", i),
                "logical_index": i,   # 0-based subtitle track index for ffmpeg -map
                "codec":    s.get("codec_name", "unknown"),
                "language": tags.get("language", "und"),
                "title":    tags.get("title", ""),
            })
        return tracks
    except Exception as e:
        print(f"[!] ffprobe parse error: {e}")
        return []


# ==========================================
# SRT PARSER
# ==========================================

def parse_srt_time(t):
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    t = t.strip().replace(',', '.')
    parts = t.split(':')
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def parse_srt(srt_text):
    """
    Parse SRT content into a list of subtitle entries.
    Returns: [{start, end, text}, ...]
    """
    entries = []
    blocks  = re.split(r'\n\s*\n', srt_text.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue

        # Find the timing line (may not be on line 2 if index is missing)
        time_line = None
        text_start = 0
        for i, line in enumerate(lines):
            if '-->' in line:
                time_line  = line
                text_start = i + 1
                break

        if time_line is None:
            continue

        try:
            parts = time_line.split('-->')
            start = parse_srt_time(parts[0])
            end   = parse_srt_time(parts[1].split()[0])  # ignore positioning tags
        except Exception:
            continue

        # Join remaining lines as text, strip HTML/ASS tags
        raw_text = ' '.join(lines[text_start:]).strip()
        text     = re.sub(r'<[^>]+>', '', raw_text)        # strip <i>, <b>, etc.
        text     = re.sub(r'\{[^}]+\}', '', text)          # strip {ASS override tags}
        text     = text.strip()

        if text:
            entries.append({"start": round(start, 3), "end": round(end, 3), "text": text})

    return entries


# ==========================================
# ASS / SSA PARSER (fallback)
# ==========================================

def parse_ass_time(t):
    """Convert ASS timestamp (H:MM:SS.cs) to seconds."""
    t = t.strip()
    parts = t.split(':')
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def parse_ass(ass_text):
    """
    Parse ASS/SSA content into subtitle entries.
    Returns: [{start, end, text}, ...]
    """
    entries = []
    in_events = False

    for line in ass_text.splitlines():
        line = line.strip()
        if line.lower() == '[events]':
            in_events = True
            continue
        if in_events and line.startswith('['):
            in_events = False

        if in_events and line.lower().startswith('dialogue:'):
            # Format: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
            parts = line.split(',', 9)
            if len(parts) < 10:
                continue
            try:
                start = parse_ass_time(parts[1])
                end   = parse_ass_time(parts[2])
            except Exception:
                continue

            raw_text = parts[9]
            # Strip ASS override tags like {\an8}, {\i1}, etc.
            text = re.sub(r'\{[^}]*\}', '', raw_text)
            text = text.replace('\\N', ' ').replace('\\n', ' ').strip()

            if text:
                entries.append({"start": round(start, 3), "end": round(end, 3), "text": text})

    return entries


# ==========================================
# EXTRACTION LOGIC
# ==========================================

def extract_subtitle_track(video_path, logical_index, output_format="srt"):
    """
    Extract a subtitle track from the video using ffmpeg.
    Returns the raw subtitle text, or None on failure.
    """
    with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = (
            f'ffmpeg -y -v error '
            f'-i "{video_path}" '
            f'-map 0:s:{logical_index} '
            f'"{tmp_path}"'
        )
        _, stderr, rc = run(cmd)

        if rc != 0:
            # Some codecs can't be exported directly as SRT — try WebVTT
            if output_format == "srt":
                return extract_subtitle_track(video_path, logical_index, output_format="vtt")
            print(f"  [!] ffmpeg error:\n{stderr}")
            return None, None

        with open(tmp_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        detected_format = output_format
        return content, detected_format

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def parse_subtitle_content(content, fmt):
    """Parse subtitle content based on its format. Returns list of entries."""
    if fmt in ("srt", "vtt"):
        # WebVTT is basically SRT-compatible for our purposes
        return parse_srt(content)
    elif fmt in ("ass", "ssa"):
        return parse_ass(content)
    else:
        # Try SRT first, then ASS
        entries = parse_srt(content)
        if not entries:
            entries = parse_ass(content)
        return entries


# ==========================================
# MAIN FLOW
# ==========================================

def extract_subtitles(video_path, output_path=None, track_index=None):
    """
    Full extraction pipeline.
    - video_path:  path to the video file
    - output_path: where to save the JSON (default: same dir as video, same name + .json)
    - track_index: which subtitle track to use (0-based); None = auto-pick
    Returns: output_path on success, None on failure.
    """

    if not os.path.isfile(video_path):
        print(f"[!] Video not found: {video_path}")
        return None

    print(f"\nVideo: {video_path}")

    # 1. List subtitle tracks
    tracks = list_subtitle_tracks(video_path)
    if not tracks:
        print("[!] No embedded subtitle tracks found in this video.")
        print("    Tip: Subtitles may be in a separate file, or hardcoded (burned-in) into the video.")
        return None

    print(f"\nFound {len(tracks)} subtitle track(s):")
    for t in tracks:
        title_part = f" — {t['title']}" if t['title'] else ""
        print(f"  [{t['logical_index']}] codec={t['codec']:<12} lang={t['language']}{title_part}")

    # 2. Pick track
    if track_index is not None:
        chosen = track_index
    elif len(tracks) == 1:
        chosen = 0
        print(f"\nAuto-selected track [0].")
    else:
        # Prefer English; otherwise first track
        eng = next((t for t in tracks if t['language'] in ('eng', 'en')), None)
        chosen = eng['logical_index'] if eng else 0
        print(f"\nAuto-selected track [{chosen}] (language preference).")
        print("  Use --track N to override.")

    chosen_track = next((t for t in tracks if t['logical_index'] == chosen), None)
    if chosen_track is None:
        print(f"[!] Track index {chosen} not found.")
        return None

    print(f"\nExtracting track [{chosen}] ({chosen_track['codec']}, {chosen_track['language']})...")

    # 3. Determine output format based on codec
    codec = chosen_track['codec'].lower()
    if codec in ('ass', 'ssa'):
        fmt = 'ass'
    else:
        fmt = 'srt'   # handles subrip, webvtt, mov_text, dvd_subtitle (best-effort)

    content, actual_fmt = extract_subtitle_track(video_path, chosen, fmt)
    if content is None:
        print("[!] Extraction failed.")
        return None

    # 4. Parse
    entries = parse_subtitle_content(content, actual_fmt)
    if not entries:
        print("[!] Subtitle content extracted but no entries could be parsed.")
        print("    Raw content preview:")
        print(content[:500])
        return None

    print(f"  Parsed {len(entries)} subtitle entries.")
    if entries:
        print(f"  First entry: [{entries[0]['start']}s - {entries[0]['end']}s] \"{entries[0]['text'][:80]}\"")
        print(f"  Last entry:  [{entries[-1]['start']}s - {entries[-1]['end']}s] \"{entries[-1]['text'][:80]}\"")

    # 5. Save JSON
    os.makedirs("subtitles", exist_ok=True)

    if output_path is None:
        base        = os.path.splitext(os.path.basename(video_path))[0]
        lang        = chosen_track['language']
        output_path = f"subtitles/{base}_{lang}.json"
    else:
        # Always save into subtitles/, strip any directory the user may have typed
        filename    = os.path.basename(output_path)
        output_path = f"subtitles/{filename}.json"


    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(entries)} entries to: {output_path}")
    return output_path


# ==========================================
# CLI
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract embedded subtitles from a video and save as JSON."
    )
    parser.add_argument(
        "--video", "-v",
        help="Path to the video file."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON name. Default: <video_name>_<lang>.json next to the video."
    )
    parser.add_argument(
        "--track", "-t",
        type=int,
        default=None,
        help="Subtitle track index to extract (0-based). Default: auto-select."
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Only list available subtitle tracks, do not extract."
    )
    args = parser.parse_args()

    # Interactive mode if no --video provided
    if not args.video:
        print("="*60)
        print("  SUBTITLE EXTRACTOR")
        print("="*60)
        while True:
            video_path = input("\nVideo file path: ").strip()
            if os.path.isfile(video_path):
                break
            print(f"  [!] File not found: {video_path}")

        output_path = input("Output JSON name (leave empty for auto): ").strip() or None
        track_input = input("Subtitle track index (leave empty for auto): ").strip()
        track_index = int(track_input) if track_input.isdigit() else None
    else:
        video_path  = args.video
        output_path = args.output
        track_index = args.track

        if args.list:
            tracks = list_subtitle_tracks(video_path)
            if not tracks:
                print("No subtitle tracks found.")
            else:
                print(f"\n{len(tracks)} subtitle track(s) in: {video_path}")
                for t in tracks:
                    title_part = f" — {t['title']}" if t['title'] else ""
                    print(f"  [{t['logical_index']}] codec={t['codec']:<12} lang={t['language']}{title_part}")
            sys.exit(0)

    result = extract_subtitles(video_path, output_path, track_index)
    if result is None:
        sys.exit(1)

    print("\nDone! Use this JSON as the subtitle path in the Video Explorer:")
    print(f"  Subtitle JSON: {result}")


if __name__ == "__main__":
    main()

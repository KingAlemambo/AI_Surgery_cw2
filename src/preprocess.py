from pathlib import Path
import os
import cv2
import pandas as pd
from collections import defaultdict

# -------------------------
# Paths
# -------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "cholec80"

# -------------------------
# Phase mapping (match dataset exactly)
# -------------------------
PHASE_TO_ID = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6,
}

# -------------------------
# Frame extraction (1 fps)
# -------------------------
def extract_frames_1fps(video_path, output_dir, fps=25):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % fps == 0:
            filename = f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(output_dir / filename), frame)

        frame_id += 1

    cap.release()

# -------------------------
# Load annotations
# -------------------------
def load_phase_annotations(path):
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["Frame"], df["Phase"]))

def load_tool_annotations(path):
    df = pd.read_csv(path, sep="\t")
    tool_cols = df.columns[1:]
    tools_by_frame = {}

    for _, row in df.iterrows():
        frame_id = int(row["Frame"])
        tools_by_frame[frame_id] = row[tool_cols].astype(int).tolist()

    return tools_by_frame

# -------------------------
# Build frame-level samples
# -------------------------
def build_frame_table(video_id, frames_dir, phase_by_frame, tools_by_frame, fps=25):
    samples = []
    frame_files = sorted(frames_dir.glob("*.jpg"))

    tool_frames = sorted(tools_by_frame.keys())

    for f in frame_files:
        frame_id = int(f.stem.replace("frame_", ""))
        time_sec = frame_id / fps

        phase_name = phase_by_frame[frame_id]
        phase_id = PHASE_TO_ID[phase_name]

        # Use nearest previous tool annotation if missing
        # Use nearest previous tool annotation if missing (safe)
        if frame_id in tools_by_frame:
            tools = tools_by_frame[frame_id]
        else:
            prev_candidates = [t for t in tool_frames if t <= frame_id]
            if len(prev_candidates) == 0:
                # No previous tool annotation → assume no tools present
                tools = [0] * len(next(iter(tools_by_frame.values())))
            else:
                prev = max(prev_candidates)
                tools = tools_by_frame[prev]


        samples.append({
            "video_id": video_id,
            "frame_id": frame_id,
            "time_sec": time_sec,
            "image_path": str(f),
            "phase": phase_name,
            "phase_id": phase_id,
            "tools": tools,
        })

    return samples

# -------------------------
# Add time targets (Task A)
# -------------------------
def add_time_targets(samples):
    # Remaining surgery time
    T_end = samples[-1]["time_sec"]
    for s in samples:
        s["t_surgery_remaining"] = T_end - s["time_sec"]

    # Remaining phase time
    start = 0
    current_phase = samples[0]["phase_id"]

    for i in range(1, len(samples)):
        if samples[i]["phase_id"] != current_phase:
            end = i - 1
            end_time = samples[end]["time_sec"]
            for j in range(start, end + 1):
                samples[j]["t_phase_remaining"] = end_time - samples[j]["time_sec"]

            start = i
            current_phase = samples[i]["phase_id"]

    # Final phase
    end_time = samples[-1]["time_sec"]
    for j in range(start, len(samples)):
        samples[j]["t_phase_remaining"] = end_time - samples[j]["time_sec"]

    return samples

# -------------------------
# Main entry
# -------------------------
def preprocess_video(video_id):
    video_path = DATA_DIR / "videos" / f"{video_id}.mp4"
    phase_txt  = DATA_DIR / "phase_annotations" / f"{video_id}-phase.txt"
    tool_txt   = DATA_DIR / "tool_annotations" / f"{video_id}-tool.txt"
    frames_dir = DATA_DIR / "frames" / video_id

    if not frames_dir.exists() or len(list(frames_dir.glob("*.jpg"))) == 0:
        print(f"Extracting frames for {video_id}")
        extract_frames_1fps(video_path, frames_dir)
    else:
        print(f"Frames already exist for {video_id}, skipping extraction")
        
    phase_by_frame = load_phase_annotations(phase_txt)
    tools_by_frame = load_tool_annotations(tool_txt)

    samples = build_frame_table(
        video_id,
        frames_dir,
        phase_by_frame,
        tools_by_frame
    )

    samples = add_time_targets(samples)
    return samples

if __name__ == "__main__":
    samples = preprocess_video("video01")
    print(f"Preprocessed {len(samples)} samples for video01")

import cv2
import os
import pandas as pd
from pathlib import Path


# maps phases to integers so that we can trian the network
PHASE_TO_ID = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6,
}
ROOT_DIR = Path(__file__).resolve().parents[1]  # project root
DATA_DIR = ROOT_DIR / "data" / "cholec80"
# extract frames per second of each video
def extract_frames_1fps(video_path, output_dir, fps=25):
    """
    Extract frames at 1 fps from a video and save them as JPG files.
    Frame IDs are preserved in filenames.
    """

    os.makedirs(output_dir, exist_ok= True)

    cap = cv2.VideoCapture(video_path)

    # keeps track of the original frame number
    frame_id =0

    while True:
        # ret shows if a frame was read, and frame store the actual image as a numpy array
        ret, frame = cap.read()
        if not ret:
            break

        # keep only every 25th frame (1fps)
        if frame_id % fps == 0:
            filename = f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)

        frame_id +=1

    cap.release()


def load_phase_annotations(phase_txt_path):
    """
    Reads videoXX-phase.txt and returns:
    { frame_id : phase_name }
    """
    df = pd.read_csv(phase_txt_path, sep = "\t")
    phase_by_frame = dict(zip(df["Frame"], df["Phase"]))

    return phase_by_frame

def load_tool_annotations(tool_tx_path):
    """
    Reads videoXX-tool.txt and returns:
    { frame_id : [7 binary tool values] }
    """
    df = pd.read_csv(tool_tx_path, sep ="\t" )
    
    tool_cols_names = df.columns[1:] # ship first column of frame
    tools_by_frame = {}

    for _, row in df.iterrows():
        frame_id = int(row["Frame"])
        tools_by_frame[frame_id] = row[tool_cols_names].astype(int).tolist()

    return tools_by_frame

def build_frame_table(
        video_id,
        frames_dir,
        phase_by_frame,
        tools_by_frame,
        fps=25
):
    """
    Builds the Step 1 dataset:
    list of dictionaries, one per sampled frame.
    """
    samples = []

    frame_files = sorted(os.listdir(frames_dir))

    for fname in frame_files:
        frame_id = int(fname.replace("frame_", "").replace(".jpg", ""))
        # convert into seconds
        time_sec = frame_id / fps

        phase_name = phase_by_frame[frame_id]
        phase_id = PHASE_TO_ID[phase_name]

        if frame_id in tools_by_frame:
            tools = tools_by_frame[frame_id]
        else:
            # Use the nearest previous tool annotation
            prev_frames = [f for f in tools_by_frame.keys() if f < frame_id]
            if not prev_frames:
                raise ValueError(f"No tool annotation found before frame {frame_id}")
            nearest_frame = max(prev_frames)
            tools = tools_by_frame[nearest_frame]


        sample = {
            "video_id": video_id,
            "frame_id": frame_id,
            "time_sec": time_sec,
            "image_path": os.path.join(frames_dir, fname),
            "phase": phase_name,
            "phase_id": phase_id,
            "tools": tools
        }

        samples.append(sample)

    return samples

def add_time_targets(samples):
    """
    Adds Task A time targets to each sample dict in-place:
      - t_surgery_remaining (seconds)
      - t_phase_remaining (seconds)

    Assumes:
      - samples are in time order (they are, because frames_dir listing is sorted)
      - time_sec increases by 1 each sample (1 fps sampling)
    """

    if not samples:
        raise ValueError("No samples provided to add_time_targets.")
    
    T_end = samples[-1]["time_sec"]  # last sampled second
    for s in samples:
        s["t_surgery_remaining"] = float(T_end - s["time_sec"])

    # --- B) Remaining phase time ---
    # Walk through the samples and find contiguous phase segments.
    start_idx = 0
    current_phase = samples[0]["phase_id"]

    for i in range(1, len(samples)):
        phase_i = samples[i]["phase_id"]

        # When phase changes, the previous segment ends at i-1
        if phase_i != current_phase:
            end_idx = i - 1
            seg_end_time = samples[end_idx]["time_sec"]

            # Assign remaining phase time for all samples in this segment
            for j in range(start_idx, end_idx + 1):
                samples[j]["t_phase_remaining"] = float(seg_end_time - samples[j]["time_sec"])

            # Start new segment
            start_idx = i
            current_phase = phase_i

    # Handle final segment (runs to end of video)
    end_idx = len(samples) - 1
    seg_end_time = samples[end_idx]["time_sec"]
    for j in range(start_idx, end_idx + 1):
        samples[j]["t_phase_remaining"] = float(seg_end_time - samples[j]["time_sec"])

    return samples



if __name__ == "__main__":
    print("MAIN BLOCK EXECUTED")

    video_id = "video01"

    video_path = DATA_DIR / "videos" / "video01.mp4"
    phase_txt  = DATA_DIR / "phase_annotations" / "video01-phase.txt"
    tool_txt   = DATA_DIR / "tool_annotations" / "video01-tool.txt"
    frames_dir = DATA_DIR / "frames" / "video01"

    print("Video exists:", video_path.exists())
    print("Phase txt exists:", phase_txt.exists())
    print("Tool txt exists:", tool_txt.exists())

    # 1. Extract frames
    extract_frames_1fps(video_path, frames_dir)

    print("Frame files:", len(list(frames_dir.glob("*.jpg"))))

    # 2. Load annotations
    phase_by_frame = load_phase_annotations(phase_txt)
    tools_by_frame = load_tool_annotations(tool_txt)

    print("Phase annotations:", len(phase_by_frame))
    print("Tool annotations:", len(tools_by_frame))

    # 3. Build dataset
    samples = build_frame_table(
        video_id,
        frames_dir,
        phase_by_frame,
        tools_by_frame
    )
    samples = add_time_targets(samples)


    print("Samples built:", len(samples))

    # 4. Inspect
    for s in samples[:5]:
        print({
        "frame_id": s["frame_id"],
        "time_sec": s["time_sec"],
        "phase": s["phase"],
        "phase_id": s["phase_id"],
        "tools": s["tools"],
        "t_phase_remaining": s["t_phase_remaining"],
        "t_surgery_remaining": s["t_surgery_remaining"],
    })
        print("\nSamples around FIRST phase change:")

    # Find first phase change to inspect boundary behavior
    first_change = None
    for i in range(1, len(samples)):
        if samples[i]["phase_id"] != samples[i-1]["phase_id"]:
            first_change = i
            break

    if first_change is not None:
        print("\nSamples around FIRST phase change (boundary check):")
        start = max(0, first_change - 3)
        end = min(len(samples), first_change + 3)
        for s in samples[start:end]:
            print({
                "time_sec": s["time_sec"],
                "phase": s["phase"],
                "phase_id": s["phase_id"],
                "t_phase_remaining": s["t_phase_remaining"],
            })
    else:
        print("\nNo phase change found in this video (unexpected but possible for short clips).")


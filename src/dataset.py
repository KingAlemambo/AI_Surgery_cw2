import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
import torchvision.transforms as T
from preprocess import preprocess_video


# Normalization constant for elapsed time
# Most cholecystectomies are under 60 minutes, so we normalize by this
MAX_SURGERY_DURATION_MIN = 60.0


class Cholec80TimeDataset(Dataset):
    """
    PyTorch Dataset for time prediction in Cholec80.

    Each item returns:
      - image sequence: Tensor [T, 3, H, W]
      - targets at current time:
          * t_phase_remaining: minutes until current phase ends
          * t_surgery_remaining: minutes until surgery ends
          * phase_id: current surgical phase (0-6)
          * tools: binary vector of tool presence
          * progress: fraction of surgery completed (0-1)
          * elapsed_time: [T, 1] tensor of normalized elapsed times for LSTM input
    """
    def __init__(self, samples, sequence_length=30, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform

        # Group samples by video
        self.samples_by_video = defaultdict(list)
        for s in samples:
            self.samples_by_video[s["video_id"]].append(s)

        # Sort samples by time within each video
        for vid in self.samples_by_video:
            self.samples_by_video[vid] = sorted(
                self.samples_by_video[vid],
                key=lambda x: x["time_sec"]
            )

        # Pre-compute total duration for each video (for progress calculation)
        self.video_duration = {}
        for vid, seq in self.samples_by_video.items():
            self.video_duration[vid] = seq[-1]["time_sec"]

        # Build index of valid (video, end_frame) pairs
        self.index = []
        for vid, seq in self.samples_by_video.items():
            for i in range(sequence_length - 1, len(seq)):
                self.index.append((vid, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        vid, end_idx = self.index[idx]
        seq = self.samples_by_video[vid]
        total_duration = self.video_duration[vid]

        start_idx = end_idx - self.sequence_length + 1
        window = seq[start_idx:end_idx + 1]

        # Load and transform images
        images = []
        elapsed_times = []

        for s in window:
            # Load image
            img = Image.open(s["image_path"]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

            # Collect elapsed time for each frame in sequence
            # Normalize by max expected duration (60 min)
            elapsed_min = s["time_sec"] / 60.0
            elapsed_normalized = elapsed_min / MAX_SURGERY_DURATION_MIN
            elapsed_times.append(elapsed_normalized)

        images = torch.stack(images)  # [T, 3, H, W]

        # Elapsed time tensor for LSTM input: [T, 1]
        elapsed_time_tensor = torch.tensor(elapsed_times, dtype=torch.float32).unsqueeze(-1)

        # Current frame targets (last frame in window)
        current = seq[end_idx]

        # Compute progress: what fraction of surgery is complete?
        # This is a self-supervised signal (no manual annotation needed)
        progress = current["time_sec"] / total_duration if total_duration > 0 else 0.0

        targets = {
            # Time remaining predictions (in minutes)
            "t_phase_remaining": torch.tensor(
                current["t_phase_remaining"] / 60.0, dtype=torch.float32
            ),
            "t_surgery_remaining": torch.tensor(
                current["t_surgery_remaining"] / 60.0, dtype=torch.float32
            ),
            # Phase classification
            "phase_id": torch.tensor(current["phase_id"], dtype=torch.long),
            # Tool presence (for Task B)
            "tools": torch.tensor(current["tools"], dtype=torch.float32),
            # Progress: self-supervised signal (0-1)
            "progress": torch.tensor(progress, dtype=torch.float32),
            # Elapsed time sequence for LSTM input
            "elapsed_time": elapsed_time_tensor,
            # Task A: Start and end times for ALL phases (in minutes)
            # [7] values each - one per surgical phase
            "phase_start_remaining": torch.tensor(
                [t / 60.0 for t in current["phase_start_remaining"]], dtype=torch.float32
            ),
            "phase_end_remaining": torch.tensor(
                [t / 60.0 for t in current["phase_end_remaining"]], dtype=torch.float32
            ),
        }

        return images, targets

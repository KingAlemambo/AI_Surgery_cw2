import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
import torchvision.transforms as T
from preprocess import preprocess_video



class Cholec80TimeDataset(Dataset):
    """
    PyTorch Dataset for time prediction in Cholec80.

    Each item returns:
      - image sequence: Tensor [T, 3, H, W]
      - targets at current time:
          * t_phase_remaining
          * t_surgery_remaining
          * phase_id 
          * tools 
    """
    def __init__(self, samples, sequence_length=30, transform = None):

        self.sequence_length = sequence_length
        self.transform = transform


        # structuring and grouping sampels by video
        self.samples_by_video = defaultdict(list)
        for s in samples:
            self.samples_by_video[s["video_id"]].append(s)

        # sorting samples by time
        for vid in self.samples_by_video:
            self.samples_by_video[vid] =  sorted(
                self.samples_by_video[vid],
                key = lambda x: x["time_sec"]
                )

        # indexing
        self.index = []
        for vid, seq in self.samples_by_video.items():
            for i in range(sequence_length - 1, len(seq)):
                self.index.append((vid, i))
    
    # returns the number of training exmaples 
    def __len__(self):
        return len(self.index)
    

    def __getitem__(self, idx):

        vid , end_idx = self.index[idx]

        seq = self.samples_by_video[vid]

        start_idx = end_idx - self.sequence_length+ 1
        # tale only the elements within this window
        window = seq[start_idx:end_idx + 1]

        images = []
        # window represents a training sample (one surgical frame with all the metadata)
        # we convert these windows into tensor
        for s in window:
            img = Image.open(s["image_path"]).convert("RGB")

            # conversion needed (images to a short video clip)
            if self.transform:
                img = self.transform(img)
            
            images.append(img)

        images = torch.stack(images)

        # data of the last frame captured in the window
        # represent present ground truth
        current_target = seq[end_idx]

        # target is a learning signal
        targets = {
            # seconds remaining in the current phase
            "t_phase_remaining": torch.tensor(
                current_target["t_phase_remaining"] / 60.0, dtype=torch.float32
            ),
            # seconds remaining in the entire surgery
            "t_surgery_remaining": torch.tensor(
                current_target["t_surgery_remaining"] / 60.0, dtype=torch.float32
            ),
            # surgical phase we are in 
            "phase_id": torch.tensor(
                current_target["phase_id"] , dtype=torch.long
            ),
            # tools present at the current second
            "tools": torch.tensor(
                current_target["tools"], dtype=torch.float32
            ),
        }

        return images, targets






"""
Task B: Tool Detection Experiments

This script runs the baseline tool detection experiment:
- Model: CNN + LSTM with multi-task (tools + phase)
- No time features (baseline for comparison)

Later we will add the timed version to compare.

Usage:
    python main_taskb.py
"""

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path

from models.cnn import ResNet50_FeatureExtractor
from models.tool_detector import ToolDetectorBaseline, ToolDetectorTimed, CHOLEC80_TOOLS
from dataset import Cholec80TimeDataset
from train_tools import train
from preprocess import preprocess_video
from utils import split_by_video

# -------------------------
# Constants
# -------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "cholec80"
VIDEOS_DIR = DATA_DIR / "videos"

# ImageNet normalization (required for pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(is_training=True):
    """
    Get image transforms for training or validation.
    Same as Task A for consistency.
    """
    if is_training:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def run_baseline_experiment(train_samples, val_samples, sequence_length=30):
    """
    Run the baseline tool detection experiment (no time features).

    This is the first experiment for Task B:
    - CNN + LSTM architecture
    - Multi-task: tool detection + phase classification
    - No time information (will be added in timed version)
    """
    print("\n" + "=" * 80)
    print("TASK B - BASELINE EXPERIMENT: Tool Detection (No Time Features)")
    print("=" * 80)

    # Create datasets
    train_dataset = Cholec80TimeDataset(
        samples=train_samples,
        sequence_length=sequence_length,
        transform=get_transforms(is_training=True)
    )

    val_dataset = Cholec80TimeDataset(
        samples=val_samples,
        sequence_length=sequence_length,
        transform=get_transforms(is_training=False)
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    cnn = ResNet50_FeatureExtractor(pretrained=True, freeze=False)
    model = ToolDetectorBaseline(cnn=cnn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup checkpoint path
    ckpt_dir = Path("checkpoints_taskb")
    ckpt_dir.mkdir(exist_ok=True)
    checkpoint_path = ckpt_dir / "baseline_tool_detector.pt"

    # Train
    history = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        epochs=20,
        batch_size=8,  # Increased for faster training
        checkpoint_path=str(checkpoint_path)
    )

    # Plot results
    plot_results(history, "baseline")

    return history


def run_timed_experiment(train_samples, val_samples, sequence_length=30):
    """
    Run the timed tool detection experiment (WITH time features).

    This is the comparison experiment for Task B:
    - CNN + LSTM architecture
    - Multi-task: tool detection + phase classification
    - WITH time information (elapsed_time as input)

    Hypothesis: Knowing where we are in surgery helps predict which tools are present.
    """
    print("\n" + "=" * 80)
    print("TASK B - TIMED EXPERIMENT: Tool Detection (WITH Time Features)")
    print("=" * 80)

    # Create datasets (same as baseline)
    train_dataset = Cholec80TimeDataset(
        samples=train_samples,
        sequence_length=sequence_length,
        transform=get_transforms(is_training=True)
    )

    val_dataset = Cholec80TimeDataset(
        samples=val_samples,
        sequence_length=sequence_length,
        transform=get_transforms(is_training=False)
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model - ToolDetectorTimed instead of Baseline
    cnn = ResNet50_FeatureExtractor(pretrained=True, freeze=False)
    model = ToolDetectorTimed(cnn=cnn, num_time_features=1)  # 1 time feature: elapsed_time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup checkpoint path
    ckpt_dir = Path("checkpoints_taskb")
    ckpt_dir.mkdir(exist_ok=True)
    checkpoint_path = ckpt_dir / "timed_tool_detector.pt"

    # Train with use_time=True
    history = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        epochs=20,
        batch_size=8,
        checkpoint_path=str(checkpoint_path),
        use_time=True  # KEY DIFFERENCE: pass time features to model
    )

    # Plot results
    plot_results(history, "timed")

    return history


def plot_results(history, exp_name):
    """
    Plot training curves for tool detection.
    """
    plots_dir = Path("plots_taskb")
    plots_dir.mkdir(exist_ok=True)

    epochs = range(1, len(history["train_mAP"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: mAP
    axes[0].plot(epochs, history["train_mAP"], label="Train", marker='o')
    axes[0].plot(epochs, history["val_mAP"], label="Val", marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("mAP")
    axes[0].set_title(f"Tool Detection mAP ({exp_name})")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Phase Accuracy
    axes[1].plot(epochs, history["train_phase_acc"], label="Train", marker='o')
    axes[1].plot(epochs, history["val_phase_acc"], label="Val", marker='s')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Phase Classification Accuracy ({exp_name})")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = plots_dir / f"{exp_name}_training_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved plot to {plot_path}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    # Find all video files
    VIDEO_IDS = sorted([p.stem for p in VIDEOS_DIR.glob("*.mp4")])
    print(f"Found {len(VIDEO_IDS)} videos")

    # Preprocess all videos (extract frames, load annotations)
    print("Loading and preprocessing videos...")
    all_samples = []
    for vid in VIDEO_IDS:
        all_samples.extend(preprocess_video(vid))

    print(f"Total samples: {len(all_samples)}")

    # Split by video (not by frame) to prevent data leakage
    train_samples, val_samples, test_samples = split_by_video(all_samples)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Subsample for faster training (use every 3rd sample)
    # This reduces ~123k samples to ~41k, making epochs ~3x faster
    SUBSAMPLE_FACTOR = 3
    train_samples = train_samples[::SUBSAMPLE_FACTOR]
    val_samples = val_samples[::SUBSAMPLE_FACTOR]
    print(f"After subsampling (1/{SUBSAMPLE_FACTOR}): Train: {len(train_samples)}, Val: {len(val_samples)}")

    # ==========================================
    # Run Baseline Experiment (already completed - skip)
    # ==========================================
    # baseline_history = run_baseline_experiment(
    #     train_samples=train_samples,
    #     val_samples=val_samples,
    #     sequence_length=30
    # )

    # ==========================================
    # Run Timed Experiment
    # ==========================================
    timed_history = run_timed_experiment(
        train_samples=train_samples,
        val_samples=val_samples,
        sequence_length=30  # Same as baseline for fair comparison
    )

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 80)
    print("TASK B TIMED RESULTS")
    print("=" * 80)
    print(f"Best Validation mAP: {max(timed_history['val_mAP']):.3f}")
    print(f"Best Validation Phase Acc: {max(timed_history['val_phase_acc']):.3f}")
    print("\nTimed training complete!")
    print("\nCompare with Baseline: mAP=0.960, Phase Acc=0.892")

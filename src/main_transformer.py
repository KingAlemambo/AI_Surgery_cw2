"""
Task A: Transformer vs LSTM Comparison

This script runs the Transformer model for duration prediction
with sequence length 30 (same as best LSTM) for comparison.

Usage:
    python main_transformer.py
"""

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path

from models.cnn import ResNet50_FeatureExtractor
from models.cnn_transformer import CNNTransformerPhaseModel
from dataset import Cholec80TimeDataset
from train import train
from preprocess import preprocess_video
from utils import split_by_video

# -------------------------
# Constants
# -------------------------
MIN_TO_SEC = 60.0

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "cholec80"
VIDEOS_DIR = DATA_DIR / "videos"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(is_training=True):
    """Get image transforms for training or validation."""
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


def run_transformer_experiment(train_samples, val_samples, sequence_length=30):
    """
    Run Transformer experiment for Task A.

    Uses same configuration as LSTM for fair comparison:
    - Sequence length: 30
    - d_model: 256 (same as LSTM hidden_dim)
    - Unfrozen Layer4
    """
    print("\n" + "=" * 70)
    print("TASK A - TRANSFORMER EXPERIMENT")
    print(f"Sequence Length: {sequence_length}, d_model: 256, nhead: 8, layers: 2")
    print("=" * 70)

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
    model = CNNTransformerPhaseModel(
        cnn=cnn,
        d_model=256,  # Same as LSTM hidden_dim
        nhead=8,
        num_layers=2  # Same as LSTM
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup checkpoint path
    ckpt_dir = Path("checkpoints_transformer")
    ckpt_dir.mkdir(exist_ok=True)
    checkpoint_path = ckpt_dir / f"transformer_seq{sequence_length}_best.pt"

    # Train
    metrics = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        epochs=20,
        batch_size=4,
        checkpoint_path=str(checkpoint_path)
    )

    # Convert MAE from minutes to seconds
    val_phase_mae_sec = [m * MIN_TO_SEC for m in metrics["val_phase_mae"]]
    val_surgery_mae_sec = [m * MIN_TO_SEC for m in metrics["val_surgery_mae"]]

    best_val_phase_mae_sec = min(val_phase_mae_sec)
    best_val_surgery_mae_sec = min(val_surgery_mae_sec)

    # Create plots
    plots_dir = Path("plots_transformer")
    plots_dir.mkdir(exist_ok=True)

    epochs_range = range(1, len(val_phase_mae_sec) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, [m * MIN_TO_SEC for m in metrics["train_phase_mae"]], label="Train", marker='o')
    axes[0].plot(epochs_range, val_phase_mae_sec, label="Val", marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE (seconds)")
    axes[0].set_title("Transformer - Phase Remaining Time")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, [m * MIN_TO_SEC for m in metrics["train_surgery_mae"]], label="Train", marker='o')
    axes[1].plot(epochs_range, val_surgery_mae_sec, label="Val", marker='s')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (seconds)")
    axes[1].set_title("Transformer - Surgery Remaining Time")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = plots_dir / f"transformer_seq{sequence_length}_mae.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\nSaved plot to {plot_path}")
    print(f"Best Val Phase MAE: {best_val_phase_mae_sec:.1f} sec ({best_val_phase_mae_sec/60:.2f} min)")
    print(f"Best Val Surgery MAE: {best_val_surgery_mae_sec:.1f} sec ({best_val_surgery_mae_sec/60:.2f} min)")

    return {
        "best_val_phase_mae_min": best_val_phase_mae_sec / 60,
        "best_val_surgery_mae_min": best_val_surgery_mae_sec / 60,
        "metrics": metrics
    }


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    # Find all video files
    VIDEO_IDS = sorted([p.stem for p in VIDEOS_DIR.glob("*.mp4")])
    print(f"Found {len(VIDEO_IDS)} videos")

    # Preprocess all videos
    print("Loading and preprocessing videos...")
    all_samples = []
    for vid in VIDEO_IDS:
        all_samples.extend(preprocess_video(vid))

    print(f"Total samples: {len(all_samples)}")

    # Split by video
    train_samples, val_samples, test_samples = split_by_video(all_samples)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Run Transformer experiment with seq=30
    results = run_transformer_experiment(
        train_samples=train_samples,
        val_samples=val_samples,
        sequence_length=30
    )

    # Summary comparison with LSTM results
    print("\n" + "=" * 70)
    print("COMPARISON: LSTM vs Transformer (Task A, Sequence Length = 30)")
    print("=" * 70)
    print(f"LSTM (from previous experiments):")
    print(f"  Phase MAE:   7.27 min")
    print(f"  Surgery MAE: 13.80 min")
    print(f"\nTransformer (this experiment):")
    print(f"  Phase MAE:   {results['best_val_phase_mae_min']:.2f} min")
    print(f"  Surgery MAE: {results['best_val_surgery_mae_min']:.2f} min")

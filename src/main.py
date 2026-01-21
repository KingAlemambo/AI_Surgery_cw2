import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path

from models.cnn import ResNet50_FeatureExtractor
from models.cnn_lstm import CNNLSTMPhaseModel
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

# ImageNet normalization values (required for pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(is_training=True):
    """
    Get image transforms for training or validation.

    For training:
    - Random augmentation to improve generalization
    - Helps model learn invariance to brightness, contrast, flips

    For validation:
    - Deterministic transforms only
    - Ensures consistent evaluation

    Both use:
    - 224x224 size (ResNet standard input size)
    - ImageNet normalization (required because ResNet was pretrained on ImageNet)
    """
    if is_training:
        return T.Compose([
            T.Resize((256, 256)),        # Resize larger first
            T.RandomCrop(224),           # Then random crop to 224
            T.RandomHorizontalFlip(p=0.5),  # 50% chance to flip
            T.ColorJitter(
                brightness=0.2,          # Random brightness ±20%
                contrast=0.2,            # Random contrast ±20%
                saturation=0.1,          # Random saturation ±10%
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),        # Direct resize for validation
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


# -------------------------
# Run one experiment
# -------------------------
def run_experiment(train_samples, val_samples, sequence_length, exp_name, freeze_cnn):
    """
    Run a single training experiment with specified configuration.

    Args:
        train_samples: List of training sample dictionaries
        val_samples: List of validation sample dictionaries
        sequence_length: Number of frames in each input sequence
        exp_name: Name for this experiment (used for checkpoints/plots)
        freeze_cnn: If True, freeze all CNN layers. If False, train layer4.
    """
    print(f"\n{'='*60}")
    print(f"Running {exp_name} | Sequence Length = {sequence_length}")
    print(f"{'='*60}")

    # Create datasets with appropriate transforms
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
    cnn = ResNet50_FeatureExtractor(
        pretrained=True,
        freeze=freeze_cnn
    )

    model = CNNLSTMPhaseModel(cnn=cnn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Debug: confirm CNN trainability
    cnn_trainable = sum(p.numel() for p in model.cnn.parameters() if p.requires_grad)
    cnn_total = sum(p.numel() for p in model.cnn.parameters())
    print(f"CNN trainable params: {cnn_trainable:,} / {cnn_total:,} ({100*cnn_trainable/cnn_total:.1f}%)")

    # Setup checkpoint path
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    checkpoint_path = ckpt_dir / f"{exp_name}_seq{sequence_length}_best.pt"

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

    # Convert MAE from minutes to seconds for reporting
    train_phase_mae_sec = [m * MIN_TO_SEC for m in metrics["train_phase_mae"]]
    val_phase_mae_sec = [m * MIN_TO_SEC for m in metrics["val_phase_mae"]]
    train_surgery_mae_sec = [m * MIN_TO_SEC for m in metrics["train_surgery_mae"]]
    val_surgery_mae_sec = [m * MIN_TO_SEC for m in metrics["val_surgery_mae"]]

    best_val_phase_mae_sec = min(val_phase_mae_sec)
    best_val_surgery_mae_sec = min(val_surgery_mae_sec)

    # Create plots
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    epochs_range = range(1, len(train_phase_mae_sec) + 1)

    # Plot 1: Phase MAE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, train_phase_mae_sec, label="Train", marker='o')
    axes[0].plot(epochs_range, val_phase_mae_sec, label="Val", marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE (seconds)")
    axes[0].set_title(f"{exp_name} – Phase Remaining Time")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, train_surgery_mae_sec, label="Train", marker='o')
    axes[1].plot(epochs_range, val_surgery_mae_sec, label="Val", marker='s')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (seconds)")
    axes[1].set_title(f"{exp_name} – Surgery Remaining Time")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = plots_dir / f"{exp_name}_seq{sequence_length}_mae.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved plot to {plot_path}")
    print(f"Best Val Phase MAE: {best_val_phase_mae_sec:.1f} sec ({best_val_phase_mae_sec/60:.2f} min)")
    print(f"Best Val Surgery MAE: {best_val_surgery_mae_sec:.1f} sec ({best_val_surgery_mae_sec/60:.2f} min)")

    return {
        "sequence_length": sequence_length,
        "best_val_phase_mae_sec": best_val_phase_mae_sec,
        "best_val_surgery_mae_sec": best_val_surgery_mae_sec,
        "checkpoint_path": str(checkpoint_path),
        "train_phase_mae_sec": train_phase_mae_sec,
        "val_phase_mae_sec": val_phase_mae_sec,
    }


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

    # ===============================
    # Experiment: Compare frozen vs unfrozen CNN
    # ===============================
    results_all = []

    # Test with sequence length 30 (30 seconds of context)
    seq_len = 30

    # Experiment 1: Frozen CNN (only LSTM trains)
    results_frozen = run_experiment(
        train_samples=train_samples,
        val_samples=val_samples,
        sequence_length=seq_len,
        exp_name="Exp_FrozenCNN",
        freeze_cnn=True
    )
    results_all.append(("Frozen CNN", results_frozen))

    # Experiment 2: Unfrozen layer4 (CNN adapts to surgical domain)
    results_unfrozen = run_experiment(
        train_samples=train_samples,
        val_samples=val_samples,
        sequence_length=seq_len,
        exp_name="Exp_UnfrozenL4",
        freeze_cnn=False
    )
    results_all.append(("Unfrozen Layer4", results_unfrozen))

    # ===============================
    # Summary
    # ===============================
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for name, r in results_all:
        print(f"{name:20s} | Phase MAE: {r['best_val_phase_mae_sec']:6.1f}s | Surgery MAE: {r['best_val_surgery_mae_sec']:6.1f}s")

    # Comparison plot
    plt.figure(figsize=(10, 6))
    for name, r in results_all:
        epochs = range(1, len(r['val_phase_mae_sec']) + 1)
        plt.plot(epochs, r['val_phase_mae_sec'], label=name, marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Validation Phase MAE (seconds)")
    plt.title("Frozen vs Unfrozen CNN Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path("plots") / "frozen_vs_unfrozen_comparison.png", dpi=150)
    plt.close()

    print("\nTraining complete!")

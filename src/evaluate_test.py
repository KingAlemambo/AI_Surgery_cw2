"""
Test Set Evaluation and Visualization for Task A & Task B

This script:
1. Evaluates Task A (LSTM) on test set
2. Evaluates Task B (baseline, elapsed, predicted) on test set
3. Generates combined visualization figure

Usage:
    python evaluate_test.py
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm

# Local imports
from models.cnn import ResNet50_FeatureExtractor
from models.cnn_lstm import CNNLSTMPhaseModel
from models.tool_detector import ToolDetectorBaseline, ToolDetectorTimed
from dataset import Cholec80TimeDataset
from preprocess import preprocess_video
from utils import split_by_video

# -------------------------
# Constants
# -------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "cholec80"
VIDEOS_DIR = DATA_DIR / "videos"

# Model checkpoint paths - ADJUST THESE TO YOUR ACTUAL PATHS
TASK_A_CHECKPOINT = "checkpoints5/Exp_Seq30_seq30_best.pt"  # Best LSTM model
TASK_B_BASELINE_CHECKPOINT = "checkpoints_taskb/baseline_best.pt"
TASK_B_TIMED_CHECKPOINT = "checkpoints_taskb/timed_best.pt"  # elapsed time
TASK_B_PREDICTED_CHECKPOINT = "checkpoints_taskb/timed_predicted_best.pt"  # predicted time

TOOL_NAMES = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]


def get_transforms():
    """Validation/test transforms (no augmentation)."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


# -------------------------
# Task A Evaluation
# -------------------------
@torch.no_grad()
def evaluate_task_a(model, dataloader, device):
    """Evaluate Task A model on test set."""
    model.eval()

    time_loss_fn = nn.L1Loss(reduction='none')

    all_phase_mae = []
    all_surgery_mae = []
    all_phase_starts_mae = []
    all_phase_ends_mae = []
    correct = 0
    total = 0

    for images, targets in tqdm(dataloader, desc="Evaluating Task A"):
        images = images.to(device)
        t_phase_gt = targets["t_phase_remaining"].to(device)
        t_surgery_gt = targets["t_surgery_remaining"].to(device)
        elapsed_time = targets["elapsed_time"].to(device)
        phase_labels = targets["phase_id"].to(device)
        phase_starts_gt = targets["phase_start_remaining"].to(device)
        phase_ends_gt = targets["phase_end_remaining"].to(device)

        outputs = model(images, elapsed_time)

        # Collect per-sample MAE
        phase_mae = time_loss_fn(outputs["t_phase_pred"], t_phase_gt)
        surgery_mae = time_loss_fn(outputs["t_surgery_pred"], t_surgery_gt)
        starts_mae = time_loss_fn(outputs["phase_starts_pred"], phase_starts_gt).mean(dim=1)
        ends_mae = time_loss_fn(outputs["phase_ends_pred"], phase_ends_gt).mean(dim=1)

        all_phase_mae.extend(phase_mae.cpu().numpy())
        all_surgery_mae.extend(surgery_mae.cpu().numpy())
        all_phase_starts_mae.extend(starts_mae.cpu().numpy())
        all_phase_ends_mae.extend(ends_mae.cpu().numpy())

        # Phase accuracy
        preds = outputs["phase_logits"].argmax(dim=1)
        correct += (preds == phase_labels).sum().item()
        total += phase_labels.size(0)

    results = {
        "phase_mae": np.mean(all_phase_mae),
        "phase_mae_std": np.std(all_phase_mae),
        "surgery_mae": np.mean(all_surgery_mae),
        "surgery_mae_std": np.std(all_surgery_mae),
        "phase_starts_mae": np.mean(all_phase_starts_mae),
        "phase_ends_mae": np.mean(all_phase_ends_mae),
        "phase_acc": correct / total
    }

    return results


# -------------------------
# Task B Evaluation
# -------------------------
@torch.no_grad()
def evaluate_task_b(model, dataloader, device, use_time=False, task_a_model=None):
    """Evaluate Task B model on test set."""
    model.eval()

    all_tool_preds = []
    all_tool_targets = []
    phase_correct = 0
    phase_total = 0

    for images, targets in tqdm(dataloader, desc="Evaluating Task B"):
        images = images.to(device)
        phase_labels = targets["phase_id"].to(device)
        tool_labels = targets["tools"].to(device).float()

        if use_time:
            if task_a_model is not None:
                # Use predicted time from Task A
                elapsed_time = targets["elapsed_time"].to(device)
                task_a_outputs = task_a_model(images, elapsed_time)
                predicted_time = task_a_outputs["t_surgery_pred"]
                B, T = images.shape[0], images.shape[1]
                time_features = predicted_time.unsqueeze(1).unsqueeze(2).expand(B, T, 1)
            else:
                # Use ground truth elapsed time
                time_features = targets["elapsed_time"].to(device)
            outputs = model(images, time_features)
        else:
            outputs = model(images)

        # Collect predictions
        tool_probs = torch.sigmoid(outputs["tool_logits"]).cpu().numpy()
        all_tool_preds.extend(tool_probs)
        all_tool_targets.extend(tool_labels.cpu().numpy())

        # Phase accuracy
        phase_preds = outputs["phase_logits"].argmax(dim=1)
        phase_correct += (phase_preds == phase_labels).sum().item()
        phase_total += phase_labels.size(0)

    # Compute mAP
    predictions = np.array(all_tool_preds)
    targets_arr = np.array(all_tool_targets)

    per_tool_ap = []
    for i in range(predictions.shape[1]):
        if targets_arr[:, i].sum() > 0:
            ap = average_precision_score(targets_arr[:, i], predictions[:, i])
            per_tool_ap.append(ap)
        else:
            per_tool_ap.append(0.0)

    mAP = np.mean(per_tool_ap)

    return {
        "mAP": mAP,
        "per_tool_ap": per_tool_ap,
        "phase_acc": phase_correct / phase_total
    }


# -------------------------
# Visualization
# -------------------------
def create_combined_figure(task_a_results, task_b_results, save_path="plots/test_results_summary.png"):
    """Create combined visualization for both tasks."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Task A: Duration Prediction ---
    ax1 = axes[0]

    metrics = ['Phase\nRemaining', 'Surgery\nRemaining', 'Phase\nStarts', 'Phase\nEnds']
    values = [
        task_a_results['phase_mae'],
        task_a_results['surgery_mae'],
        task_a_results['phase_starts_mae'],
        task_a_results['phase_ends_mae']
    ]

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    bars1 = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)

    ax1.set_ylabel('MAE (minutes)', fontsize=12)
    ax1.set_title('Task A: Duration Prediction\n(Test Set)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(values) * 1.2)

    # Add value labels on bars
    for bar, val in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Task B: Tool Detection ---
    ax2 = axes[1]

    models = ['Baseline\n(no time)', 'Elapsed\n(ground truth)', 'Predicted\n(from Task A)']
    mAP_values = [
        task_b_results['baseline']['mAP'],
        task_b_results['elapsed']['mAP'],
        task_b_results['predicted']['mAP']
    ]

    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars2 = ax2.bar(models, mAP_values, color=colors, edgecolor='black', linewidth=1.2)

    ax2.set_ylabel('mAP', fontsize=12)
    ax2.set_title('Task B: Tool Detection\n(Test Set)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.9, 1.0)  # Zoom in to show differences

    # Add value labels on bars
    for bar, val in zip(bars2, mAP_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add horizontal line at baseline
    ax2.axhline(y=mAP_values[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved figure to: {save_path}")


def print_results(task_a_results, task_b_results):
    """Print formatted results."""
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)

    print("\n--- Task A: Duration Prediction (LSTM, T=30) ---")
    print(f"  Phase Remaining MAE:   {task_a_results['phase_mae']:.2f} min")
    print(f"  Surgery Remaining MAE: {task_a_results['surgery_mae']:.2f} min")
    print(f"  Phase Starts MAE:      {task_a_results['phase_starts_mae']:.2f} min")
    print(f"  Phase Ends MAE:        {task_a_results['phase_ends_mae']:.2f} min")
    print(f"  Phase Accuracy:        {task_a_results['phase_acc']:.1%}")

    print("\n--- Task B: Tool Detection ---")
    print(f"  Baseline (no time):     mAP = {task_b_results['baseline']['mAP']:.3f}, Phase Acc = {task_b_results['baseline']['phase_acc']:.1%}")
    print(f"  + Elapsed time (GT):    mAP = {task_b_results['elapsed']['mAP']:.3f}, Phase Acc = {task_b_results['elapsed']['phase_acc']:.1%}")
    print(f"  + Predicted time (A):   mAP = {task_b_results['predicted']['mAP']:.3f}, Phase Acc = {task_b_results['predicted']['phase_acc']:.1%}")

    print("\n--- Per-Tool AP (Baseline) ---")
    for name, ap in zip(TOOL_NAMES, task_b_results['baseline']['per_tool_ap']):
        print(f"  {name:12s}: {ap:.3f}")

    print("="*70)


# -------------------------
# Main
# -------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    print("\nLoading video data...")
    VIDEO_IDS = sorted([p.stem for p in VIDEOS_DIR.glob("*.mp4")])
    print(f"Found {len(VIDEO_IDS)} videos")

    all_samples = []
    for vid in VIDEO_IDS:
        all_samples.extend(preprocess_video(vid))

    train_samples, val_samples, test_samples = split_by_video(all_samples)
    print(f"Test samples: {len(test_samples)}")

    # Create test dataset
    test_dataset = Cholec80TimeDataset(
        samples=test_samples,
        sequence_length=30,
        transform=get_transforms()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )

    # -------------------------
    # Task A Evaluation
    # -------------------------
    print("\n" + "="*50)
    print("Evaluating Task A...")
    print("="*50)

    # Load Task A model
    cnn_a = ResNet50_FeatureExtractor(pretrained=True, freeze=False)
    task_a_model = CNNLSTMPhaseModel(cnn=cnn_a)

    checkpoint_a = torch.load(TASK_A_CHECKPOINT, map_location=device)
    task_a_model.load_state_dict(checkpoint_a)
    task_a_model.to(device)
    task_a_model.eval()

    task_a_results = evaluate_task_a(task_a_model, test_loader, device)

    # -------------------------
    # Task B Evaluation
    # -------------------------
    print("\n" + "="*50)
    print("Evaluating Task B...")
    print("="*50)

    task_b_results = {}

    # Baseline (no time)
    print("\nEvaluating baseline...")
    cnn_b = ResNet50_FeatureExtractor(pretrained=True, freeze=False)
    baseline_model = ToolDetectorBaseline(cnn=cnn_b)

    checkpoint_b = torch.load(TASK_B_BASELINE_CHECKPOINT, map_location=device)
    baseline_model.load_state_dict(checkpoint_b["model_state_dict"])
    baseline_model.to(device)

    task_b_results['baseline'] = evaluate_task_b(baseline_model, test_loader, device, use_time=False)

    # Elapsed time (ground truth)
    print("\nEvaluating with elapsed time...")
    cnn_e = ResNet50_FeatureExtractor(pretrained=True, freeze=False)
    elapsed_model = ToolDetectorTimed(cnn=cnn_e)

    checkpoint_e = torch.load(TASK_B_TIMED_CHECKPOINT, map_location=device)
    elapsed_model.load_state_dict(checkpoint_e["model_state_dict"])
    elapsed_model.to(device)

    task_b_results['elapsed'] = evaluate_task_b(elapsed_model, test_loader, device, use_time=True, task_a_model=None)

    # Predicted time (from Task A)
    print("\nEvaluating with predicted time...")
    cnn_p = ResNet50_FeatureExtractor(pretrained=True, freeze=False)
    predicted_model = ToolDetectorTimed(cnn=cnn_p)

    checkpoint_p = torch.load(TASK_B_PREDICTED_CHECKPOINT, map_location=device)
    predicted_model.load_state_dict(checkpoint_p["model_state_dict"])
    predicted_model.to(device)

    # Freeze Task A model for inference
    for param in task_a_model.parameters():
        param.requires_grad = False

    task_b_results['predicted'] = evaluate_task_b(predicted_model, test_loader, device, use_time=True, task_a_model=task_a_model)

    # -------------------------
    # Results
    # -------------------------
    print_results(task_a_results, task_b_results)

    # Create visualization
    create_combined_figure(task_a_results, task_b_results, save_path="plots/test_results_summary.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

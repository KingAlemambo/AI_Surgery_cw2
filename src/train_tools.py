"""
Training script for Task B: Tool Detection

This handles:
- Multi-task training (tools + phase)
- Class weighting for tool imbalance
- Metrics: mAP for tools, accuracy for phase
- Early stopping based on validation mAP
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from tqdm import tqdm


def compute_class_weights(dataset, num_tools=7):
    """
    Compute pos_weight for BCEWithLogitsLoss to handle tool class imbalance.

    Formula: pos_weight = (num_negative) / (num_positive)
    This gives higher weight to rare tools like SpecimenBag.

    Args:
        dataset: Dataset with tool labels (uses index to access raw samples)
        num_tools: Number of tools (default 7 for Cholec80)

    Returns:
        torch.Tensor: pos_weight for each tool [num_tools]
    """
    tool_counts = torch.zeros(num_tools)
    total_samples = 0

    # Fast path: access raw sample data without loading images
    # The dataset.index contains (video_id, end_idx) pairs
    # We access the underlying samples directly
    for vid, end_idx in dataset.index:
        sample = dataset.samples_by_video[vid][end_idx]
        tools = torch.tensor(sample["tools"], dtype=torch.float32)
        tool_counts += tools
        total_samples += 1

    # pos_weight = (total - positive) / positive = negative / positive
    pos_weight = (total_samples - tool_counts) / (tool_counts + 1e-6)

    # Clip to reasonable range to avoid extreme weights
    pos_weight = torch.clamp(pos_weight, min=0.5, max=20.0)

    print(f"Tool frequencies: {(tool_counts / total_samples * 100).numpy().round(1)}%")
    print(f"Pos weights: {pos_weight.numpy().round(2)}")

    return pos_weight


def compute_tool_metrics(predictions, targets, threshold=0.5):
    """
    Compute metrics for multi-label tool detection.

    Args:
        predictions: Sigmoid probabilities [N, num_tools]
        targets: Binary labels [N, num_tools]
        threshold: Classification threshold

    Returns:
        dict with mAP, per-tool AP, F1, precision, recall
    """
    predictions = np.array(predictions)
    targets = np.array(targets)

    num_tools = predictions.shape[1]

    # Per-tool Average Precision
    per_tool_ap = []
    for i in range(num_tools):
        # Skip if no positive samples for this tool
        if targets[:, i].sum() > 0:
            ap = average_precision_score(targets[:, i], predictions[:, i])
            per_tool_ap.append(ap)
        else:
            per_tool_ap.append(0.0)

    # mAP (mean Average Precision) - primary metric for multi-label
    mAP = np.mean(per_tool_ap)

    # Binary predictions for F1, precision, recall
    binary_preds = (predictions > threshold).astype(int)

    # Micro-averaged metrics (treats all predictions equally)
    f1_micro = f1_score(targets, binary_preds, average='micro', zero_division=0)
    precision_micro = precision_score(targets, binary_preds, average='micro', zero_division=0)
    recall_micro = recall_score(targets, binary_preds, average='micro', zero_division=0)

    # Macro-averaged metrics (average per tool)
    f1_macro = f1_score(targets, binary_preds, average='macro', zero_division=0)

    return {
        "mAP": mAP,
        "per_tool_ap": per_tool_ap,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro
    }


def train_one_epoch(model, dataloader, optimizer, device, tool_pos_weight=None):
    """
    Train for one epoch.

    Args:
        model: ToolDetectorBaseline or ToolDetectorTimed
        dataloader: Training DataLoader
        optimizer: Optimizer
        device: cuda or cpu
        tool_pos_weight: Class weights for tool BCE loss

    Returns:
        dict with average losses and metrics
    """
    model.train()

    # Loss functions
    phase_loss_fn = nn.CrossEntropyLoss()
    if tool_pos_weight is not None:
        tool_loss_fn = nn.BCEWithLogitsLoss(pos_weight=tool_pos_weight.to(device))
    else:
        tool_loss_fn = nn.BCEWithLogitsLoss()

    # Accumulators
    total_loss = 0.0
    total_phase_loss = 0.0
    total_tool_loss = 0.0
    num_batches = 0

    # For metrics
    all_tool_preds = []
    all_tool_targets = []
    phase_correct = 0
    phase_total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        images, targets = batch

        # Move data to device
        images = images.to(device)
        phase_labels = targets["phase_id"].to(device)
        tool_labels = targets["tools"].to(device).float()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        phase_preds = outputs["phase_logits"]
        tool_preds = outputs["tool_logits"]

        # Compute losses
        loss_phase = phase_loss_fn(phase_preds, phase_labels)
        loss_tools = tool_loss_fn(tool_preds, tool_labels)

        # Total loss (equal weighting)
        loss = loss_phase + loss_tools

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_phase_loss += loss_phase.item()
        total_tool_loss += loss_tools.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        # Track metrics
        with torch.no_grad():
            # Tool predictions (sigmoid probabilities)
            tool_probs = torch.sigmoid(tool_preds).cpu().numpy()
            all_tool_preds.extend(tool_probs)
            all_tool_targets.extend(tool_labels.cpu().numpy())

            # Phase accuracy
            phase_pred_classes = phase_preds.argmax(dim=1)
            phase_correct += (phase_pred_classes == phase_labels).sum().item()
            phase_total += phase_labels.size(0)

    # Compute metrics
    tool_metrics = compute_tool_metrics(all_tool_preds, all_tool_targets)
    phase_acc = phase_correct / phase_total if phase_total > 0 else 0.0

    return {
        "loss": total_loss / num_batches,
        "phase_loss": total_phase_loss / num_batches,
        "tool_loss": total_tool_loss / num_batches,
        "mAP": tool_metrics["mAP"],
        "f1_micro": tool_metrics["f1_micro"],
        "phase_acc": phase_acc
    }


@torch.no_grad()
def validate_one_epoch(model, dataloader, device, tool_pos_weight=None):
    """
    Validate for one epoch.

    Args:
        model: ToolDetectorBaseline or ToolDetectorTimed
        dataloader: Validation DataLoader
        device: cuda or cpu
        tool_pos_weight: Class weights for tool BCE loss

    Returns:
        dict with average losses and metrics
    """
    model.eval()

    # Loss functions
    phase_loss_fn = nn.CrossEntropyLoss()
    if tool_pos_weight is not None:
        tool_loss_fn = nn.BCEWithLogitsLoss(pos_weight=tool_pos_weight.to(device))
    else:
        tool_loss_fn = nn.BCEWithLogitsLoss()

    # Accumulators
    total_loss = 0.0
    total_phase_loss = 0.0
    total_tool_loss = 0.0
    num_batches = 0

    # For metrics
    all_tool_preds = []
    all_tool_targets = []
    phase_correct = 0
    phase_total = 0

    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for batch in pbar:
        images, targets = batch

        # Move data to device
        images = images.to(device)
        phase_labels = targets["phase_id"].to(device)
        tool_labels = targets["tools"].to(device).float()

        # Forward pass
        outputs = model(images)
        phase_preds = outputs["phase_logits"]
        tool_preds = outputs["tool_logits"]

        # Compute losses
        loss_phase = phase_loss_fn(phase_preds, phase_labels)
        loss_tools = tool_loss_fn(tool_preds, tool_labels)
        loss = loss_phase + loss_tools

        # Track losses
        total_loss += loss.item()
        total_phase_loss += loss_phase.item()
        total_tool_loss += loss_tools.item()
        num_batches += 1

        # Track metrics
        tool_probs = torch.sigmoid(tool_preds).cpu().numpy()
        all_tool_preds.extend(tool_probs)
        all_tool_targets.extend(tool_labels.cpu().numpy())

        phase_pred_classes = phase_preds.argmax(dim=1)
        phase_correct += (phase_pred_classes == phase_labels).sum().item()
        phase_total += phase_labels.size(0)

    # Compute metrics
    tool_metrics = compute_tool_metrics(all_tool_preds, all_tool_targets)
    phase_acc = phase_correct / phase_total if phase_total > 0 else 0.0

    return {
        "loss": total_loss / num_batches,
        "phase_loss": total_phase_loss / num_batches,
        "tool_loss": total_tool_loss / num_batches,
        "mAP": tool_metrics["mAP"],
        "per_tool_ap": tool_metrics["per_tool_ap"],
        "f1_micro": tool_metrics["f1_micro"],
        "f1_macro": tool_metrics["f1_macro"],
        "precision_micro": tool_metrics["precision_micro"],
        "recall_micro": tool_metrics["recall_micro"],
        "phase_acc": phase_acc
    }


def train(model, train_dataset, val_dataset, device, epochs=20, batch_size=4,
          lr=1e-4, checkpoint_path="checkpoint.pt", patience=5):
    """
    Main training loop with early stopping.

    Args:
        model: ToolDetectorBaseline or ToolDetectorTimed
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device: cuda or cpu
        epochs: Maximum epochs
        batch_size: Batch size
        lr: Learning rate
        checkpoint_path: Path to save best model
        patience: Early stopping patience

    Returns:
        dict with training history
    """
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")

    # Create DataLoaders
    # num_workers=4 for parallel data loading (much faster)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True
    )

    # Compute class weights for tool imbalance
    print("\nComputing class weights for tool imbalance...")
    tool_pos_weight = compute_class_weights(train_dataset)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # History tracking
    history = {
        "train_loss": [], "val_loss": [],
        "train_mAP": [], "val_mAP": [],
        "train_phase_acc": [], "val_phase_acc": [],
        "train_f1": [], "val_f1": []
    }

    # Early stopping
    best_val_mAP = 0.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, tool_pos_weight
        )

        # Validate
        val_metrics = validate_one_epoch(
            model, val_loader, device, tool_pos_weight
        )

        # Update scheduler
        scheduler.step(val_metrics["mAP"])

        # Track history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_mAP"].append(train_metrics["mAP"])
        history["val_mAP"].append(val_metrics["mAP"])
        history["train_phase_acc"].append(train_metrics["phase_acc"])
        history["val_phase_acc"].append(val_metrics["phase_acc"])
        history["train_f1"].append(train_metrics["f1_micro"])
        history["val_f1"].append(val_metrics["f1_micro"])

        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"mAP: {train_metrics['mAP']:.3f}, "
              f"Phase Acc: {train_metrics['phase_acc']:.3f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"mAP: {val_metrics['mAP']:.3f}, "
              f"Phase Acc: {val_metrics['phase_acc']:.3f}")
        print(f"Val   - F1 micro: {val_metrics['f1_micro']:.3f}, "
              f"Precision: {val_metrics['precision_micro']:.3f}, "
              f"Recall: {val_metrics['recall_micro']:.3f}")

        # Print per-tool AP
        tool_names = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
        print("\nPer-tool AP:")
        for i, (name, ap) in enumerate(zip(tool_names, val_metrics["per_tool_ap"])):
            print(f"  {name}: {ap:.3f}")

        # Check for improvement
        if val_metrics["mAP"] > best_val_mAP:
            best_val_mAP = val_metrics["mAP"]
            epochs_no_improve = 0

            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mAP": best_val_mAP,
                "history": history
            }, checkpoint_path)
            print(f"\n*** New best model saved! Val mAP: {best_val_mAP:.3f} ***")
        else:
            epochs_no_improve += 1
            print(f"\nNo improvement for {epochs_no_improve} epochs (best: {best_val_mAP:.3f})")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    print(f"\n{'='*60}")
    print(f"Training complete! Best Val mAP: {best_val_mAP:.3f}")
    print(f"{'='*60}")

    return history

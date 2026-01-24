import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from pathlib import Path

# train th emodel for one pass over the dataset
def train_one_epoch(model, dataloader, optimizer, device):
    # identify the current stage that we are training 
    model.train()
    # define the loss
    #criterion = nn.CrossEntropyLoss()
    phase_loss_fn = nn.CrossEntropyLoss()
    time_loss_fn = nn.L1Loss()
    
    running_loss = 0
    correct = 0
    total = 0
    total_phase_mae = 0
    total_surgery_mae = 0
    total_progress_mae = 0
    total_phase_starts_mae = 0
    total_phase_ends_mae = 0

    # iterate over the batches
    for batch in dataloader:
        images, targets = batch
        images = images.to(device)

        # Get all targets and move to device
        phase_labels = targets["phase_id"].to(device)
        t_phase_gt = targets["t_phase_remaining"].to(device)
        t_surgery_gt = targets["t_surgery_remaining"].to(device)
        progress_gt = targets["progress"].to(device)
        elapsed_time = targets["elapsed_time"].to(device)  # [B, T, 1]
        # Task A: All phase start/end times
        phase_starts_gt = targets["phase_start_remaining"].to(device)  # [B, 7]
        phase_ends_gt = targets["phase_end_remaining"].to(device)      # [B, 7]

        # clear gradients to zero reset
        optimizer.zero_grad()

        # Forward pass - now includes elapsed time for temporal reasoning
        outputs = model(images, elapsed_time)

        # Extract predictions
        phase_logits = outputs["phase_logits"]
        t_phase_pred = outputs["t_phase_pred"]
        t_surgery_pred = outputs["t_surgery_pred"]
        progress_pred = outputs["progress_pred"]
        phase_starts_pred = outputs["phase_starts_pred"]  # [B, 7]
        phase_ends_pred = outputs["phase_ends_pred"]      # [B, 7]

        # Compute all losses
        loss_phase = phase_loss_fn(phase_logits, phase_labels)
        loss_t_phase = time_loss_fn(t_phase_pred, t_phase_gt)
        loss_t_surgery = time_loss_fn(t_surgery_pred, t_surgery_gt)
        loss_progress = time_loss_fn(progress_pred, progress_gt)
        # Task A: Loss for all phase start/end times
        loss_phase_starts = time_loss_fn(phase_starts_pred, phase_starts_gt)
        loss_phase_ends = time_loss_fn(phase_ends_pred, phase_ends_gt)

        # Track MAE for each task
        total_phase_mae += loss_t_phase.item()
        total_surgery_mae += loss_t_surgery.item()
        total_progress_mae += loss_progress.item()
        total_phase_starts_mae += loss_phase_starts.item()
        total_phase_ends_mae += loss_phase_ends.item()

        # Combined loss with all components weighted
        # Phase classification helps model understand surgical workflow
        # Progress prediction provides self-supervised signal
        # Time predictions are the main objective
        # Task A phase times get equal weight to surgery time
        loss = (0.2 * loss_phase +
                0.15 * loss_t_phase +
                0.25 * loss_t_surgery +
                0.1 * loss_progress +
                0.15 * loss_phase_starts +  # Task A
                0.15 * loss_phase_ends)     # Task A

        # backpropagation
        loss.backward()

        # using the computed gradients to update weights 
        optimizer.step()

        # update metrics 
        running_loss += loss.item()
        preds = phase_logits.argmax(dim =1)
        correct += (preds == phase_labels).sum().item()
        total += phase_labels.size(0)

    # return the average loss for this epoch, and the accuracy
    return {
        "loss": running_loss / len(dataloader),
        "phase_mae": total_phase_mae / len(dataloader),
        "surgery_mae": total_surgery_mae / len(dataloader),
        "progress_mae": total_progress_mae / len(dataloader),
        "phase_starts_mae": total_phase_starts_mae / len(dataloader),
        "phase_ends_mae": total_phase_ends_mae / len(dataloader),
        "train_acc": correct / total
    } 

@torch.no_grad()
def validate_one_epoch(model, dataloader, device):
    model.eval()

    time_loss_fn = nn.L1Loss()
    total_phase_mae = 0
    total_surgery_mae = 0
    total_progress_mae = 0
    total_phase_starts_mae = 0
    total_phase_ends_mae = 0
    correct = 0
    total = 0
    n = 0

    for images, targets in dataloader:
        images = images.to(device)
        t_phase_gt = targets["t_phase_remaining"].to(device)
        t_surgery_gt = targets["t_surgery_remaining"].to(device)
        progress_gt = targets["progress"].to(device)
        elapsed_time = targets["elapsed_time"].to(device)
        phase_labels = targets["phase_id"].to(device)
        phase_starts_gt = targets["phase_start_remaining"].to(device)
        phase_ends_gt = targets["phase_end_remaining"].to(device)

        outputs = model(images, elapsed_time)
        t_phase_pred = outputs["t_phase_pred"]
        t_surgery_pred = outputs["t_surgery_pred"]
        progress_pred = outputs["progress_pred"]
        phase_logits = outputs["phase_logits"]
        phase_starts_pred = outputs["phase_starts_pred"]
        phase_ends_pred = outputs["phase_ends_pred"]

        total_phase_mae += time_loss_fn(t_phase_pred, t_phase_gt).item()
        total_surgery_mae += time_loss_fn(t_surgery_pred, t_surgery_gt).item()
        total_progress_mae += time_loss_fn(progress_pred, progress_gt).item()
        total_phase_starts_mae += time_loss_fn(phase_starts_pred, phase_starts_gt).item()
        total_phase_ends_mae += time_loss_fn(phase_ends_pred, phase_ends_gt).item()

        # Track phase accuracy
        preds = phase_logits.argmax(dim=1)
        correct += (preds == phase_labels).sum().item()
        total += phase_labels.size(0)
        n += 1

    return {
        "phase_mae": total_phase_mae / n,
        "surgery_mae": total_surgery_mae / n,
        "progress_mae": total_progress_mae / n,
        "phase_starts_mae": total_phase_starts_mae / n,
        "phase_ends_mae": total_phase_ends_mae / n,
        "val_acc": correct / total
    }






# train for multiple epochs 
def train(model, train_dataset, val_dataset, device, epochs = 20, batch_size =4, checkpoint_path="checkpoints/best_model.pt"):
    train_phase_mae_history = []
    val_phase_mae_history = []

    train_surgery_mae_history = []
    val_surgery_mae_history = []

    # create the dataloader that organizes the batches
    # group the dataset into batches
    train_loader = DataLoader(
        train_dataset, 
        batch_size = batch_size,
        shuffle = True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size= batch_size,
        shuffle= False,
        num_workers= 4
    )

    
    # to save best model
    best_val_mae = float("inf")
    # early stopping criteria
    patience = 3
    patience_counter = 0
   
    # define the optimizer operating on model paramters
    # Weight decay (L2 regularization) penalizes large weights,
    # preventing the model from fitting noise in training data
    cnn_params = [p for p in model.cnn.parameters() if p.requires_grad]
    other_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("cnn.") and p.requires_grad
    ]

    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": 1e-4, "weight_decay": 1e-4},
        {"params": cnn_params, "lr": 1e-5, "weight_decay": 1e-5},
    ])

     # create checkpoint directory
    ckpt_dir = Path("checkpoints5")
    ckpt_dir.mkdir(exist_ok=True)
    # repeating the training process 
    for epoch in range(epochs):
        train_metrics  = train_one_epoch(
            model, train_loader, optimizer, device)
        

        val_metrics = validate_one_epoch(
            model, val_loader, device
        )
        train_phase_mae_history.append(train_metrics["phase_mae"])
        train_surgery_mae_history.append(train_metrics["surgery_mae"])

        val_phase_mae_history.append(val_metrics["phase_mae"])
        val_surgery_mae_history.append(val_metrics["surgery_mae"])

        # Use surgery MAE for early stopping (main objective)
        val_mae = val_metrics["surgery_mae"]
        print(
            f"Epoch {epoch+1:02d} | "
            f"Phase MAE: {train_metrics['phase_mae']:.2f}/{val_metrics['phase_mae']:.2f} | "
            f"Surgery MAE: {train_metrics['surgery_mae']:.2f}/{val_metrics['surgery_mae']:.2f} | "
            f"PhaseStarts MAE: {train_metrics['phase_starts_mae']:.2f}/{val_metrics['phase_starts_mae']:.2f} | "
            f"PhaseEnds MAE: {train_metrics['phase_ends_mae']:.2f}/{val_metrics['phase_ends_mae']:.2f} | "
            f"Acc: {train_metrics['train_acc']:.2f}/{val_metrics['val_acc']:.2f}"
        )

        # EARLY STOPPING CHECK
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0

            torch.save(
                model.state_dict(),
               checkpoint_path
            )
            print(" New best model saved")

        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(" Early stopping triggered")
                break
    return {
    "train_phase_mae": train_phase_mae_history,
    "val_phase_mae": val_phase_mae_history,
    "train_surgery_mae": train_surgery_mae_history,
    "val_surgery_mae": val_surgery_mae_history,
    }
        

            

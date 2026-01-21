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

    # iterate over the batches
    for batch in dataloader:
        images, targets = batch 
        images = images.to(device)

        # operate on the same device
        #images = images.to(device)
       # labels = targets["phase_id"].to(device)
        phase_labels = targets["phase_id"].to(device)
        t_phase_gt = targets["t_phase_remaining"].to(device)
        t_surgery_gt = targets["t_surgery_remaining"].to(device)

        # clear gradients to zero reset
        optimizer.zero_grad()
        outputs = model(images)

        # forward pass calling the CNN
        phase_logits = outputs["phase_logits"]
        t_phase_pred = outputs["t_phase_pred"]
        t_surgery_pred = outputs["t_surgery_pred"]
        # compute loss
        # by comparing prediction with truth
        #loss = criterion(logits, labels)

        loss_phase = phase_loss_fn(phase_logits, phase_labels)
        loss_t_phase = time_loss_fn(t_phase_pred, t_phase_gt)
        loss_t_surgery = time_loss_fn(t_surgery_pred, t_surgery_gt)
        total_phase_mae += time_loss_fn(t_phase_pred, t_phase_gt).item()
        total_surgery_mae += time_loss_fn(t_surgery_pred, t_surgery_gt).item()


        loss = loss = loss_t_phase + loss_t_surgery

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
    "train_acc": correct / total
    } 

@torch.no_grad()
def validate_one_epoch(model, dataloader, device):
    model.eval()

    time_loss_fn = nn.L1Loss()
    total_phase_mae = 0
    total_surgery_mae = 0
    n = 0


    for images, targets in dataloader:
        images = images.to(device)
        t_phase_gt = targets["t_phase_remaining"].to(device)
        t_surgery_gt = targets["t_surgery_remaining"].to(device)


        outputs = model(images)
        t_phase_pred = outputs["t_phase_pred"]
        t_surgery_pred = outputs["t_surgery_pred"]

        total_phase_mae += time_loss_fn(t_phase_pred, t_phase_gt).item()
        total_surgery_mae += time_loss_fn(t_surgery_pred, t_surgery_gt).item()
        n += 1

    return {
        "phase_mae": total_phase_mae / n,
        "surgery_mae": total_surgery_mae / n
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
    cnn_params = [p for p in model.cnn.parameters() if p.requires_grad]
    other_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("cnn.") and p.requires_grad
    ]

    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": 1e-4},
        {"params": cnn_params, "lr": 1e-5},
    ])

     # create checkpoint directory
    ckpt_dir = Path("checkpoints1")
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

        val_mae = val_metrics["phase_mae"]
        print(
         f"Epoch {epoch+1:02d} | "
        f"Train Phase MAE: {train_metrics['phase_mae']:.2f} | "
        f"Val Phase MAE: {val_metrics['phase_mae']:.2f} | "
        f"Train Surgery MAE: {train_metrics['surgery_mae']:.2f} | "
        f"Val Surgery MAE: {val_metrics['surgery_mae']:.2f}",
        f"Train Accuracy: {train_metrics['train_acc']:.2f}"
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
        

            

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()

    # define losses ONCE per epoch
    phase_loss_fn = nn.CrossEntropyLoss()
    time_loss_fn = nn.L1Loss()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        # move inputs to device
        images = images.to(device)

        phase_labels = targets["phase_id"].to(device)
        t_phase_gt = targets["t_phase_remaining"].to(device)
        t_surgery_gt = targets["t_surgery_remaining"].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        phase_logits = outputs["phase_logits"]
        t_phase_pred = outputs["t_phase_pred"]
        t_surgery_pred = outputs["t_surgery_pred"]

        loss_phase = phase_loss_fn(phase_logits, phase_labels)
        loss_t_phase = time_loss_fn(t_phase_pred, t_phase_gt)
        loss_t_surgery = time_loss_fn(t_surgery_pred, t_surgery_gt)

        loss = loss_phase + 0.1 * loss_t_phase + 0.1 * loss_t_surgery

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = phase_logits.argmax(dim=1)
        correct += (preds == phase_labels).sum().item()
        total += phase_labels.size(0)

    return running_loss / len(dataloader), correct / total


def train(model, dataset, device, epochs=10, batch_size=8):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # create checkpoint directory
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        loss, acc = train_one_epoch(
            model, dataloader, optimizer, device
        )

        print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}")

        # SAVE MODEL CHECKPOINT
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "accuracy": acc,
            },
            ckpt_dir / f"model_epoch_{epoch+1}.pt"
        )

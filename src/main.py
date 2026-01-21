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

# -------------------------
# Run one experiment
# -------------------------
def run_experiment(train_samples, val_samples, sequence_length, exp_name, freeze_cnn):

    print(f"\n=== Running {exp_name} | Seq Len = {sequence_length} ===")

    transform = T.Compose([
        T.Resize((128, 128)),  # speed-friendly baseline
        T.ToTensor()
    ])

    train_dataset = Cholec80TimeDataset(
        samples=train_samples,
        sequence_length=sequence_length,
        transform=transform
    )

    val_dataset = Cholec80TimeDataset(
        samples=val_samples,
        sequence_length=sequence_length,
        transform=transform
    )

    cnn = ResNet50_FeatureExtractor(
        pretrained=True,
        freeze=freeze_cnn
    )

    model = CNNLSTMPhaseModel(cnn=cnn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ---- DEBUG: confirm CNN trainability ----
    cnn_trainable = sum(p.numel() for p in model.cnn.parameters() if p.requires_grad)
    cnn_total = sum(p.numel() for p in model.cnn.parameters())
    print(f"CNN trainable params: {cnn_trainable} / {cnn_total}")

    ckpt_dir = Path("checkpoints2")
    ckpt_dir.mkdir(exist_ok=True)
    checkpoint_path = ckpt_dir / f"{exp_name}_seq{sequence_length}_best.pt"

    metrics = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        epochs=20,
        batch_size=4,
        checkpoint_path=str(checkpoint_path)
    )

    # ---- Convert MAE to seconds ----
    train_phase_mae_sec = [m * MIN_TO_SEC for m in metrics["train_phase_mae"]]
    val_phase_mae_sec = [m * MIN_TO_SEC for m in metrics["val_phase_mae"]]

    best_val_phase_mae_sec = min(val_phase_mae_sec)
    best_val_surgery_mae_sec = min(m * MIN_TO_SEC for m in metrics["val_surgery_mae"])

    # ---- Plot ----
    plots_dir = Path("plots2")
    plots_dir.mkdir(exist_ok=True)

    epochs = range(1, len(train_phase_mae_sec) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_phase_mae_sec, label="Train MAE (phase)")
    plt.plot(epochs, val_phase_mae_sec, label="Val MAE (phase)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (seconds)")
    plt.title(f"{exp_name} – Phase Remaining Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = plots_dir / f"{exp_name}_seq{sequence_length}_phase_mae.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot to {plot_path}")

    return {
        "sequence_length": sequence_length,
        "best_val_phase_mae_sec": best_val_phase_mae_sec,
        "best_val_surgery_mae_sec": best_val_surgery_mae_sec,
        "checkpoint_path": str(checkpoint_path),
    }

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    VIDEO_IDS = sorted([p.stem for p in VIDEOS_DIR.glob("*.mp4")])

    all_samples = []
    for vid in VIDEO_IDS:
        all_samples.extend(preprocess_video(vid))

    train_samples, val_samples, test_samples = split_by_video(all_samples)

    # ===============================
    # Experiment A2 – Frozen CNN
    # ===============================
    results_table = []

    for seq_len in [15, 30, 60]:
        results = run_experiment(
            train_samples=train_samples,
            val_samples=val_samples,
            sequence_length=seq_len,
            exp_name="ExpA2_FrozenCNN2",
            freeze_cnn=True
        )
        results_table.append(results)

    print("\n=== Experiment A2 Results (Frozen CNN) ===")
    for r in results_table:
        print(
            f"Seq {r['sequence_length']} | "
            f"Phase MAE: {r['best_val_phase_mae_sec']:.1f} sec | "
            f"Surgery MAE: {r['best_val_surgery_mae_sec']:.1f} sec"
        )

    # ---- Sequence length comparison plot ----
    seqs = [r["sequence_length"] for r in results_table]
    maes = [r["best_val_phase_mae_sec"] for r in results_table]

    plt.figure(figsize=(7, 5))
    plt.plot(seqs, maes, marker="o")
    plt.xlabel("Sequence Length (seconds)")
    plt.ylabel("Best Validation MAE (seconds)")
    plt.title("Experiment A2 – Effect of Temporal Context Length")
    plt.grid(True)
    plt.tight_layout()

    plot_path = Path("plots") / "ExpA2_seq_length_vs_phase_mae.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot to {plot_path}")

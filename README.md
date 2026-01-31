# Surgical Workflow Analysis: Duration Prediction and Tool Detection

MPHY0043 Coursework - CNN-LSTM framework for surgical workflow analysis on Cholec80.

## Project Structure

```
AI_Surgery_cw2/
├── src/
│   ├── models/
│   │   ├── cnn.py              # ResNet-50 feature extractor
│   │   ├── lstm.py             # LSTM temporal model
│   │   ├── cnn_lstm.py         # Task A: CNN-LSTM for duration prediction
│   │   ├── tool_detector.py    # Task B: Tool detection models
│   │   ├── transformer.py      # Transformer temporal model (comparison)
│   │   └── cnn_transformer.py  # CNN-Transformer variant
│   ├── dataset.py              # Cholec80 dataset loader
│   ├── preprocess.py           # Video preprocessing
│   ├── train.py                # Task A training
│   ├── train_tools.py          # Task B training
│   ├── main.py                 # Task A main script
│   ├── main_taskb.py           # Task B main script
│   ├── main_transformer.py     # Transformer comparison
│   └── evaluate_test.py        # Test set evaluation
├── report/
│   └── main.tex                # LNCS format report
└── data/
    └── cholec80/               # Dataset (not included)
```

## Requirements

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm pillow
```

## Dataset

Download Cholec80 dataset and place in `data/cholec80/`:
- `videos/` - MP4 video files (video01.mp4, video02.mp4, ...)
- `phase_annotations/` - Phase labels
- `tool_annotations/` - Tool presence labels

## Running the Code

### Task A: Duration Prediction

Train the CNN-LSTM model for duration prediction:

```bash
cd src
python main.py
```

This will:
- Train with sequence length T=30 (optimal)
- Save checkpoint to `checkpoints5/`
- Generate training plots

### Task B: Tool Detection

Train tool detection models (requires Task A checkpoint):

```bash
cd src
python main_taskb.py
```

This trains three variants:
1. Baseline (no time features)
2. With elapsed time (ground truth)
3. With predicted time (from Task A)

### Test Set Evaluation

Evaluate all models on test set and generate summary figure:

```bash
cd src
python evaluate_test.py
```

Outputs:
- Test set metrics printed to console
- Summary figure saved to `plots/test_results_summary.png`

### Transformer Comparison (Optional)

```bash
cd src
python main_transformer.py
```

## Key Results

**Task A (Test Set):**
- Phase Remaining MAE: 5.18 min
- Surgery Remaining MAE: 10.83 min

**Task B (Test Set):**
- Baseline mAP: 0.955
- Predicted time mAP: 0.962 (best)

## Author

Alexandros Mathios - University College London

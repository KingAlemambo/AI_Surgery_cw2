# MPHY0043 Coursework Report - Draft

## Title
**Temporal Context for Surgical Workflow Analysis: Duration Prediction and Tool Detection in Cholecystectomy**

---

## Abstract (150 words max)

Accurate prediction of surgical duration and tool usage is essential for operating room scheduling, resource allocation, and intraoperative decision support. We present a deep learning framework for surgical workflow analysis using the Cholec80 dataset of laparoscopic cholecystectomy videos. Our approach combines a pretrained ResNet-50 feature extractor with an LSTM for temporal modeling. For Task A, we predict remaining phase duration, remaining surgery duration, and the start/end times of all surgical phases, achieving a mean absolute error of 7.27 minutes for phase prediction and 13.80 minutes for total surgery duration. For Task B, we investigate whether estimated temporal information can improve surgical tool detection. We compare a baseline multi-task model (tool detection + phase classification) against a time-augmented version that incorporates predicted surgical progress. Our experiments demonstrate that [RESULTS PENDING - will show if time features improve tool detection]. We provide ablation studies on sequence length, CNN fine-tuning strategies, and the effect of multi-task learning.

---

## 1. Introduction

### 1.1 Clinical Motivation

Laparoscopic cholecystectomy is one of the most frequently performed surgical procedures worldwide, with over 1.2 million operations annually in the United States alone [cite]. Despite its routine nature, surgical duration varies considerably between patients due to anatomical differences, complications, and surgeon experience. Accurate prediction of surgical duration and workflow progression has significant clinical value:

- **Operating Room Scheduling**: Improved duration estimates enable better scheduling of subsequent procedures, reducing staff overtime and patient waiting times.
- **Resource Allocation**: Knowing which tools will be needed and when allows for better preparation of surgical instruments.
- **Intraoperative Decision Support**: Real-time awareness of surgical progress can alert teams to procedures running longer than expected, potentially indicating complications.

### 1.2 Related Work

Surgical workflow analysis has been extensively studied using the Cholec80 dataset [Twinanda et al., 2016]. EndoNet demonstrated that multi-task learning of phase recognition and tool detection yields mutual benefits, as tools and phases are inherently correlated (e.g., the Clipper tool appears predominantly during the ClippingCutting phase). Recent work has explored temporal models including LSTMs [cite], Temporal Convolutional Networks (TeCNO) [cite], and Transformers [cite] for capturing the sequential nature of surgery.

However, most prior work focuses on phase recognition rather than explicit duration prediction. Predicting *when* phases will start and end, and *how long* the surgery will take, remains less explored despite its practical importance.

### 1.3 Research Questions

In this work, we address the following research questions:

**RQ1**: How does temporal context length affect surgical duration prediction accuracy? We hypothesize that longer sequences provide more information but may introduce noise from older, less relevant frames.

**RQ2**: Can multi-output prediction of phase start/end times improve upon single-target duration regression? We predict not just the current phase remaining time, but the expected timing of all future phases.

**RQ3**: Does incorporating predicted temporal information (from Task A) improve downstream tool detection (Task B)? We hypothesize that knowing the surgical progress helps predict which tools are likely to be present, as certain tools are phase-specific.

### 1.4 Contributions

Our main contributions are:
1. A CNN-LSTM architecture for multi-output surgical duration prediction, including phase-specific start and end times.
2. Systematic ablation studies on sequence length and CNN fine-tuning strategies.
3. Investigation of whether predicted temporal features improve tool detection in a multi-task learning framework.

---

## 2. Methods

### 2.1 Problem Formulation

Given a sequence of video frames from an ongoing surgery, we aim to predict:

**Task A (Duration Prediction):**
- Time remaining in current phase (minutes)
- Time remaining in surgery (minutes)
- Start and end times of all 7 surgical phases relative to current time

**Task B (Tool Detection):**
- Presence/absence of 7 surgical tools (multi-label binary classification)
- Current surgical phase (multi-class classification)

### 2.2 Architecture

Our architecture follows a two-stage design: spatial feature extraction followed by temporal modeling.

**Spatial Feature Extraction (CNN):**
We use ResNet-50 pretrained on ImageNet as our backbone. The final classification layer is removed, yielding a 2048-dimensional feature vector per frame. We investigated two fine-tuning strategies:
- *Frozen*: All CNN weights fixed; only temporal model learns
- *Partial unfreezing*: Layer4 (final residual block) trainable while earlier layers remain frozen

The rationale for partial unfreezing is that early CNN layers learn generic visual features (edges, textures) that transfer well across domains, while deeper layers learn task-specific features that benefit from adaptation to surgical imagery.

**Temporal Modeling (LSTM):**
Frame features are processed by a 2-layer LSTM with hidden dimension 256. The LSTM captures temporal dependencies and surgical progression patterns. We use dropout (p=0.3) between LSTM layers for regularization.

For **Task A**, the final LSTM hidden state is passed to regression heads predicting:
- Phase remaining time (1 output)
- Surgery remaining time (1 output)
- Phase start times (7 outputs, one per phase)
- Phase end times (7 outputs, one per phase)

For **Task B**, the final hidden state feeds two classification heads:
- Tool detection head (7 outputs with sigmoid activation)
- Phase classification head (7 outputs with softmax)

### 2.3 Time-Augmented Tool Detection

For the "timed" version of Task B, we concatenate temporal features with CNN features before the LSTM:

```
input_features = [CNN_features, elapsed_time, progress]
```

Where:
- `elapsed_time`: Normalized time since surgery start
- `progress`: Fraction of surgery completed (0 to 1)

This allows the model to leverage temporal context when predicting tools.

### 2.4 Loss Functions

**Task A:**
We use L1 loss (Mean Absolute Error) for all regression targets:
```
L_A = L1(phase_remaining) + L1(surgery_remaining) + L1(phase_starts) + L1(phase_ends)
```

L1 loss is preferred over L2 as it is less sensitive to outliers and produces predictions in interpretable units (minutes).

**Task B:**
Combined multi-task loss:
```
L_B = BCE(tool_predictions, tool_labels) + CE(phase_predictions, phase_labels)
```

For tool detection, we use Binary Cross-Entropy with class-weighted pos_weight to handle tool imbalance. Tools like SpecimenBag appear in only ~6% of frames, while Grasper appears in ~60%.

### 2.5 Training Details

- **Optimizer**: AdamW with weight decay 1e-4
- **Learning rate**: 1e-4 with ReduceLROnPlateau scheduler
- **Batch size**: 8 sequences
- **Early stopping**: Patience of 5 epochs based on validation metric
- **Data augmentation**: Random crop (256→224), horizontal flip, color jitter

### 2.6 Data Preprocessing

Frames are extracted at 1 fps from the original 25 fps videos. We use ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) as required for pretrained ResNet weights.

---

## 3. Experiments

### 3.1 Dataset

We use the **Cholec80** dataset [Twinanda et al., 2016], consisting of 80 videos of laparoscopic cholecystectomy performed by 13 surgeons. The dataset includes:
- Frame-level annotations for 7 surgical phases
- Frame-level annotations for 7 surgical tools
- Video duration ranging from ~20 to ~80 minutes

**Surgical Phases:**
1. Preparation
2. Calot Triangle Dissection
3. Clipping and Cutting
4. Gallbladder Dissection
5. Gallbladder Packaging
6. Cleaning and Coagulation
7. Gallbladder Retraction

**Surgical Tools:**
Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, SpecimenBag

### 3.2 Data Split

We split the data by video (not by frame) to prevent data leakage:
- Training: 48 videos (60%)
- Validation: 16 videos (20%)
- Test: 16 videos (20%)

This ensures the model is evaluated on completely unseen surgeries.

### 3.3 Evaluation Metrics

**Task A - Duration Prediction:**
- Mean Absolute Error (MAE) in minutes
- Clinically interpretable: "predictions are off by X minutes on average"

**Task B - Tool Detection:**
- Mean Average Precision (mAP): Standard metric for multi-label classification
- Per-tool Average Precision (AP): Identifies which tools are hardest to detect
- Phase Accuracy: For the auxiliary phase classification task

### 3.4 Ablation Studies

**Sequence Length (Task A):**
We compare T ∈ {15, 30, 60} frames (at 1 fps, this corresponds to 15, 30, 60 seconds of context).

**CNN Fine-tuning (Task A):**
We compare frozen CNN vs. unfreezing Layer4.

**Time Features (Task B):**
We compare baseline (no time features) vs. timed (with elapsed time and progress).

---

## 4. Results

### 4.1 Task A: Duration Prediction

**Table 1: Effect of Sequence Length on Duration Prediction**

| Sequence Length | Phase MAE (min) | Surgery MAE (min) |
|-----------------|-----------------|-------------------|
| T = 15          | 7.48            | 14.03             |
| T = 30          | **7.27**        | **13.80**         |
| T = 60          | 7.52            | 14.48             |

The optimal sequence length is T=30 frames (30 seconds). Shorter sequences lack sufficient temporal context, while longer sequences may introduce noise from frames too far in the past.

**Clinical Interpretation:** A surgery duration prediction error of ~14 minutes is clinically meaningful for OR scheduling. For a typical 45-minute cholecystectomy, this represents ~30% uncertainty, which could be improved with additional features (e.g., patient characteristics, surgeon ID).

**Table 2: Effect of CNN Fine-tuning**

| Strategy        | Phase MAE (min) | Surgery MAE (min) |
|-----------------|-----------------|-------------------|
| Frozen CNN      | X.XX            | X.XX              |
| Unfrozen Layer4 | X.XX            | X.XX              |

[ADD RESULTS]

### 4.2 Task B: Tool Detection

**Table 3: Baseline vs. Time-Augmented Tool Detection**

| Model    | mAP   | Phase Acc | Grasper AP | Clipper AP | SpecimenBag AP |
|----------|-------|-----------|------------|------------|----------------|
| Baseline | X.XXX | X.XX      | X.XX       | X.XX       | X.XX           |
| Timed    | X.XXX | X.XX      | X.XX       | X.XX       | X.XX           |

[ADD RESULTS WHEN TRAINING COMPLETES]

**Hypothesis:** We expect time features to particularly help phase-specific tools:
- **Clipper**: Predominantly used in ClippingCutting phase
- **SpecimenBag**: Only appears in GallbladderPackaging (near end of surgery)

### 4.3 Per-Tool Analysis

[ADD FIGURE: Bar chart comparing per-tool AP for baseline vs timed]

### 4.4 Statistical Significance

To assess whether improvements are statistically significant, we perform [Wilcoxon signed-rank test / paired t-test] comparing per-video metrics between baseline and timed models.

[ADD RESULTS]

---

## 5. Discussion

### 5.1 Key Findings

[SUMMARIZE MAIN RESULTS]

### 5.2 Limitations

- **Single dataset**: Results may not generalize to other surgical procedures or institutions
- **Class imbalance**: Rare tools (SpecimenBag, Scissors) remain challenging despite class weighting
- **Temporal resolution**: 1 fps may miss rapid tool changes

### 5.3 Clinical Implications

[DISCUSS PRACTICAL VALUE]

---

## 6. Conclusion

[SUMMARIZE AND FUTURE WORK]

Future directions include:
- Transformer-based temporal modeling
- Incorporating patient-specific features (BMI, previous surgeries)
- Real-time deployment considerations

---

## References

[1] Twinanda, A.P., et al. "EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos." IEEE TMI, 2016.

[2] Czempiel, T., et al. "TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional Networks." MICCAI, 2020.

[ADD MORE REFERENCES]

---

## Appendix (if space permits)

### Training Curves
[ADD FIGURES]

### Implementation Details
- Framework: PyTorch
- Hardware: NVIDIA GPU
- Training time: ~X hours per experiment

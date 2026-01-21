# Comprehensive Guide: Surgical Duration Prediction with Deep Learning

This document explains all the changes made to improve the surgical duration prediction model, the reasoning behind each change, and the underlying machine learning concepts.

---

## Table of Contents
1. [Overview: What We're Building](#overview)
2. [The Architecture: CNN + LSTM](#architecture)
3. [Changes Made and Why](#changes)
4. [Key Machine Learning Concepts](#concepts)
5. [Training Strategy](#training)
6. [How to Run Experiments](#experiments)

---

<a name="overview"></a>
## 1. Overview: What We're Building

### The Problem
Given a video of a laparoscopic cholecystectomy (gallbladder removal surgery), predict:
1. **Remaining phase time**: How long until the current surgical phase ends?
2. **Remaining surgery time**: How long until the entire surgery ends?
3. **Current phase**: Which of the 7 surgical phases are we in?

### Why This Matters Clinically
- **OR scheduling**: Knowing remaining time helps schedule the next surgery
- **Resource allocation**: Staff can prepare for the next case
- **Patient safety**: Unusually long phases might indicate complications

### The Approach
We use a **CNN + LSTM** architecture:
- **CNN (ResNet-50)**: Looks at each video frame and extracts visual features ("what's in this image?")
- **LSTM**: Processes the sequence of features over time ("given what I've seen so far, what comes next?")

---

<a name="architecture"></a>
## 2. The Architecture: CNN + LSTM

```
Video Frames                    CNN                      LSTM                 Predictions
[B, T, 3, 224, 224]  →  ResNet-50  →  [B, T, 2048]  →  LSTM  →  [B, 256]  →  Multiple Heads
     ↑                      ↑              ↑            ↑           ↑              ↑
  B=batch size         Extracts      2048-dim      Temporal    256-dim      Phase, Time,
  T=sequence length    visual        features      reasoning   hidden       Progress
  3=RGB channels       features      per frame                 state
  224=image size
```

### Why ResNet-50?
- **Pretrained on ImageNet**: Already knows how to recognize edges, textures, shapes, objects
- **Transfer learning**: General visual knowledge transfers to surgical images
- **2048-dimensional output**: Rich feature representation

### Why LSTM?
- **Temporal dependencies**: Surgery has structure - phases follow a specific order
- **Memory**: Can remember what happened earlier in the sequence
- **Context**: "We've been dissecting for 10 minutes" is more informative than a single frame

### Multi-Task Learning
We predict multiple things simultaneously:
1. **Phase classification** (7 classes)
2. **Phase remaining time** (regression)
3. **Surgery remaining time** (regression)
4. **Progress** (0-1, regression)

Why multi-task? Each task provides a different learning signal that helps the shared representation.

---

<a name="changes"></a>
## 3. Changes Made and Why

### Change 1: Fixed the Loss Function Bug

**Original (WRONG):**
```python
loss = loss = loss_t_phase + loss_t_surgery  # phase_loss not included!
```

**Fixed:**
```python
loss = 0.3 * loss_phase + 0.2 * loss_t_phase + 0.3 * loss_t_surgery + 0.2 * loss_progress
```

**Why this matters:**
- The phase classification head was defined but NEVER trained (gradients weren't flowing to it)
- Phase recognition helps the model understand surgical workflow
- Multi-task learning: all tasks share the CNN+LSTM backbone, so training one helps others

**Loss weights explained:**
- `0.3 * loss_phase`: Phase classification is important for understanding context
- `0.3 * loss_t_surgery`: Main objective - predicting total remaining time
- `0.2 * loss_t_phase`: Secondary objective - phase-level prediction
- `0.2 * loss_progress`: Self-supervised signal (see below)

---

### Change 2: Added Progress Signal (Self-Supervised Learning)

**What is progress?**
```python
progress = elapsed_time / total_surgery_duration  # 0.0 to 1.0
```

**Why add this?**
This is a **self-supervised signal** - we get it for FREE from timestamps, no manual annotation needed.

The key insight from RSDNet paper: *"To predict remaining time, the model must understand how far along the surgery is."*

By explicitly training the model to predict progress, we force it to learn:
- How to estimate where we are in the surgery
- The relationship: `remaining_time ≈ (1 - progress) × total_duration`

**Mathematical relationship:**
```
progress = elapsed / total
remaining = total - elapsed = total × (1 - progress)
```

So if the model learns progress well, it has a strong prior for remaining time!

---

### Change 3: Added Elapsed Time as LSTM Input

**Original:**
```python
# LSTM only sees visual features
lstm_input = cnn_features  # [B, T, 2048]
```

**Fixed:**
```python
# LSTM sees visual features + elapsed time
lstm_input = concat(cnn_features, elapsed_time)  # [B, T, 2049]
```

**Why this matters:**

Imagine you're the model and you see an image of a surgeon cutting tissue. Without knowing elapsed time:
- Could be minute 5 (early in surgery)
- Could be minute 40 (late in surgery)
- Remaining time prediction would be very different!

With elapsed time, the model can learn patterns like:
- "At 10 minutes, if I see Calot's triangle dissection, expect ~25 more minutes"
- "At 40 minutes, if I see gallbladder packaging, expect ~5 more minutes"

**Normalization:**
We normalize elapsed time by 60 minutes (typical max surgery duration) to keep values in [0, 1] range, which helps training stability.

---

### Change 4: Fixed Image Size (128×128 → 224×224)

**Original:**
```python
T.Resize((128, 128))  # Too small!
```

**Fixed:**
```python
T.Resize((224, 224))  # ResNet's native size
```

**Why this matters:**

ResNet-50 was designed and trained on 224×224 images. Using 128×128:
- Loses fine surgical details (tool tips, tissue boundaries)
- Pretrained weights expect 224×224 input statistics
- Convolutional filters are optimized for this resolution

---

### Change 5: Added ImageNet Normalization

**Original:**
```python
T.ToTensor()  # Converts to [0, 1] range
```

**Fixed:**
```python
T.ToTensor(),
T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**Why this matters:**

ResNet was trained on ImageNet with specific normalization:
- Mean subtraction: Centers data around 0
- Std division: Scales to unit variance

These exact values (0.485, 0.456, 0.406 for mean) come from ImageNet statistics.

If you don't normalize:
- Pretrained weights expect normalized inputs
- Feature extraction will be suboptimal
- Like speaking a different "language" to the network

---

### Change 6: Added Data Augmentation

**Added for training:**
```python
T.RandomCrop(224),           # Spatial variation
T.RandomHorizontalFlip(),    # Left-right invariance
T.ColorJitter(brightness=0.2, contrast=0.2)  # Lighting variation
```

**Why this matters:**

Data augmentation artificially increases dataset diversity:

1. **RandomCrop**: Model sees slightly different regions
   - Prevents overfitting to exact spatial positions
   - Surgical scene position shouldn't matter

2. **RandomHorizontalFlip**: Mirrors the image horizontally
   - Surgeon could be on either side
   - Anatomy can appear mirrored
   - Doubles effective dataset size

3. **ColorJitter**: Random brightness/contrast changes
   - OR lighting varies between surgeries
   - Camera settings differ
   - Makes model robust to lighting conditions

**Important:** Augmentation is ONLY for training, not validation/test!

---

<a name="concepts"></a>
## 4. Key Machine Learning Concepts

### Transfer Learning

**What:** Using a model trained on one task (ImageNet classification) for another task (surgical analysis).

**Why it works:**
- Early CNN layers learn generic features (edges, textures)
- These features are useful for ANY visual task
- Only need to adapt later layers to new domain

**Our approach:**
- Freeze layers 1-3 (generic features)
- Optionally unfreeze layer 4 (high-level features that can adapt to surgical domain)

### Multi-Task Learning

**What:** Training one model to do multiple related tasks simultaneously.

**Why it works:**
- Tasks share a common representation (CNN + LSTM backbone)
- Learning one task provides useful gradients for others
- Acts as regularization - prevents overfitting to any single task

**Our tasks:**
1. Phase classification (categorical)
2. Phase remaining time (regression)
3. Surgery remaining time (regression)
4. Progress prediction (regression)

### Self-Supervised Learning

**What:** Learning from data that provides its own labels (no manual annotation needed).

**Our example:** Progress prediction
- Label = elapsed_time / total_duration
- Comes directly from video timestamps
- No surgeon needs to annotate anything

**Why it's powerful:**
- Free labels = can use more data
- Forces model to learn useful representations
- Progress understanding helps time prediction

### Regression vs Classification

**Classification:** Predicting a category (phase 1, 2, 3, 4, 5, 6, or 7)
- Loss: Cross-entropy
- Output: Probability distribution over classes

**Regression:** Predicting a continuous value (remaining time in minutes)
- Loss: L1 (Mean Absolute Error) or L2 (Mean Squared Error)
- Output: Single number

**Why L1 loss for time prediction?**
- L1 is more robust to outliers than L2
- A 60-minute surgery vs 30-minute surgery shouldn't dominate the loss
- MAE is interpretable: "on average, we're off by X minutes"

---

<a name="training"></a>
## 5. Training Strategy

### Optimizer Setup

```python
optimizer = torch.optim.Adam([
    {"params": other_params, "lr": 1e-4},   # LSTM, heads: higher LR
    {"params": cnn_params, "lr": 1e-5},      # CNN layer4: lower LR
])
```

**Why different learning rates?**
- LSTM and heads are randomly initialized → need larger updates
- CNN layer4 has pretrained weights → small updates to fine-tune, not destroy

### Early Stopping

```python
if val_mae < best_val_mae:
    save_model()
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= 3:
        stop_training()
```

**Why?**
- Prevents overfitting: Stop when validation performance stops improving
- Saves best model, not final model
- Patience of 3: Give it a few chances before stopping

### Batch Size Considerations

We use batch_size=4 because:
- Video sequences are large (30 frames × 224×224×3)
- GPU memory is limited
- Smaller batches = more gradient updates per epoch

---

<a name="experiments"></a>
## 6. How to Run Experiments

### After These Changes, Run:

```bash
cd AI_Surgery_cw2/src
python main.py
```

This will run two experiments:
1. **Frozen CNN**: Only LSTM trains, CNN is fixed ImageNet extractor
2. **Unfrozen Layer4**: CNN layer4 adapts to surgical domain

### Expected Improvements

| Metric | Before (estimated) | After (expected) |
|--------|-------------------|------------------|
| Surgery MAE | 400-800 sec | 300-450 sec |
| Phase MAE | Similar | Better |
| Phase Accuracy | Not trained! | 70-85% |

### Ablation Studies to Try

1. **Sequence length**: Try 15, 30, 60 frames
   - Longer = more context but slower training

2. **With/without progress**: Compare models with and without progress head
   - Shows value of self-supervised signal

3. **With/without elapsed time**: Remove elapsed_time from LSTM input
   - Shows value of explicit temporal information

4. **Frozen vs unfrozen**: Compare layer4 frozen vs trainable
   - Shows value of domain adaptation

---

## Summary of What We Fixed

| Issue | Impact | Fix |
|-------|--------|-----|
| Phase loss missing | Phase head not training | Added to total loss |
| No progress signal | Missing self-supervision | Added progress prediction |
| No elapsed time input | Model can't reason about time | Concatenate with features |
| 128×128 images | Loss of surgical detail | Use 224×224 |
| No normalization | Mismatched to pretrained weights | ImageNet normalization |
| No augmentation | Overfitting, poor generalization | Random transforms |

These changes align your implementation with the published papers (RSDNet, EndoNet) and should significantly improve performance.

from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 10, 'Surgical Duration Prediction - Deep Learning Guide', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(3)

    def chapter(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_fill_color(220, 220, 220)
        self.cell(0, 8, title, fill=True, new_x='LMARGIN', new_y='NEXT')
        self.ln(3)

    def section(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 6, title, new_x='LMARGIN', new_y='NEXT')
        self.ln(1)

    def text(self, content):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, content)
        self.ln(2)

    def code(self, content):
        self.set_font('Courier', '', 9)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(0, 4, content, fill=True)
        self.ln(2)

# Create PDF 1: Implementation Guide
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

pdf.set_font('Helvetica', 'B', 18)
pdf.cell(0, 12, 'Complete Implementation Guide', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.set_font('Helvetica', 'I', 12)
pdf.cell(0, 8, 'Surgical Duration Prediction with Deep Learning', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(8)

pdf.chapter('Part 1: The Problem We Are Solving')

pdf.section('What is the Task?')
pdf.text("""Given a video of a laparoscopic cholecystectomy (gallbladder removal surgery), we want to predict at any moment:

1. How much longer will the current phase last? (phase remaining time)
2. How much longer will the entire surgery last? (surgery remaining time)
3. Which phase are we in right now? (phase classification)""")

pdf.section('Why Is This Hard?')
pdf.text("""- Surgeries vary wildly: One cholecystectomy might take 25 minutes, another 90 minutes
- Same visual appearance, different times: A grasper tool could mean 5 min or 40 min remaining depending on context
- Single frames are ambiguous: You need to understand the sequence of events over time""")

pdf.section('Clinical Value')
pdf.text("""- OR schedulers: Know when to prepare next patient
- Surgical team: Anticipate upcoming phases
- Anesthesiologists: Plan medication timing
- Hospital admin: Optimize resource allocation""")

pdf.chapter('Part 2: Our Architecture - CNN + LSTM')

pdf.section('The Big Picture')
pdf.text("""Our model has two main components working together:

1. CNN (ResNet-50): Looks at each video frame and extracts visual features. It answers "what is in this image?" by converting raw pixels into meaningful representations.

2. LSTM: Processes the sequence of features over time. It answers "given what I have seen so far, what comes next?" by maintaining memory of past frames.""")

pdf.section('Data Flow Through the Model')
pdf.code("""Video Frames [B, 30, 3, 224, 224]
        |
        v
    ResNet-50 (CNN) --> extracts 2048-dim features per frame
        |
        v
Features [B, 30, 2048] + Elapsed Time [B, 30, 1]
        |
        v (concatenate)
Combined [B, 30, 2049]
        |
        v
    LSTM (256 hidden) --> temporal reasoning
        |
        v
    Hidden State [B, 256]
        |
        v
    4 Prediction Heads:
    - Phase classification (7 classes)
    - Phase time remaining (regression)
    - Surgery time remaining (regression)
    - Progress 0-1 (regression)""")

pdf.section('Why ResNet-50?')
pdf.text("""- Deep enough: 50 layers can learn complex hierarchical features
- Residual connections: Solves vanishing gradient problem in deep networks
- Pretrained on ImageNet: Already knows edges, textures, shapes, objects from 1.2M images
- Transfer learning: General visual knowledge transfers well to surgical images""")

pdf.section('Why LSTM?')
pdf.text("""- Temporal dependencies: Surgery has structure - phases follow specific order
- Memory: Can remember what happened earlier in the sequence
- Context: "10 minutes of dissection" is more informative than a single frame
- Sequential processing: Naturally handles variable-length sequences""")

pdf.section('Why Multi-Task Learning?')
pdf.text("""We train ONE model to predict FOUR things simultaneously. Benefits:

1. Tasks help each other: Phase recognition helps time prediction
2. Shared representation: CNN+LSTM learns features useful for all tasks
3. Regularization: Each task constrains learning, preventing overfitting
4. Efficiency: One forward pass produces multiple outputs""")

pdf.add_page()
pdf.chapter('Part 3: Key Papers and Their Contributions')

pdf.section('1. EndoNet (Twinanda et al., IEEE-TMI 2016)')
pdf.text("""Title: "EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos"

Key contributions:
- First CNN for multi-task learning on surgical videos
- Introduced Cholec80 dataset (80 cholecystectomy videos)
- Multi-task: phase recognition + tool detection together

What we used: Multi-task architecture concept, dataset structure, phase definitions""")

pdf.section('2. Aksamentov et al. (MICCAI 2017)')
pdf.text("""Title: "Deep Neural Networks Predict Remaining Surgery Duration from Cholecystectomy Videos"

Key contributions:
- CNN + LSTM architecture for time prediction
- Elapsed time as input to LSTM (crucial insight!)
- L1 loss for regression (robust to outliers)
- Achieved MAE of 7.7 minutes on Cholec120

What we used: Elapsed time input, L1 loss, CNN-LSTM pipeline architecture""")

pdf.section('3. RSDNet (Twinanda et al., IEEE-TMI 2019)')
pdf.text("""Title: "RSDNet: Learning to Predict Remaining Surgery Duration Without Manual Annotations"

Key contributions:
- Progress signal as self-supervised learning (no annotation needed!)
- Progress = elapsed_time / total_duration (free from timestamps)
- Learning progress helps learn time prediction
- Achieved MAE of 8.1 minutes on Cholec120

What we used: Progress prediction head, self-supervised signal concept""")

pdf.section('4. Less is More (Yengera et al., 2018)')
pdf.text("""Title: "Surgical Phase Recognition with Less Annotations through Self-Supervised Pre-training"

Key contributions:
- RSD prediction as pre-training task
- End-to-end CNN-LSTM training
- Shows how progress understanding transfers to phase recognition

What we used: Concept of progress as auxiliary task that helps main task""")

pdf.chapter('Part 4: Implementation Details')

pdf.section('Image Preprocessing')
pdf.text("""1. Resize to 224x224 (ResNet standard input size)
2. ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
3. Training augmentation: RandomCrop, HorizontalFlip, ColorJitter
4. No augmentation for validation (deterministic evaluation)""")

pdf.section('Loss Function')
pdf.code("""total_loss = 0.3 * phase_loss        # CrossEntropy for classification
         + 0.2 * phase_time_loss   # L1 for regression
         + 0.3 * surgery_time_loss # L1 for regression (main goal)
         + 0.2 * progress_loss     # L1 for regression (self-supervised)""")

pdf.section('Training Configuration')
pdf.text("""- Optimizer: Adam with different learning rates
  - LSTM and heads: lr = 1e-4 (learning from scratch)
  - CNN layer4: lr = 1e-5 (fine-tuning pretrained weights)
- Batch size: 4 (limited by GPU memory for video sequences)
- Early stopping: patience = 3 epochs
- Sequence length: 30 frames (30 seconds at 1fps)""")

pdf.section('Freezing Strategy')
pdf.text("""Frozen CNN: All ResNet layers frozen, only LSTM trains
- Fast training, uses ImageNet features as-is
- Good baseline, but may not capture surgical-specific features

Unfrozen Layer4: Early layers frozen, layer4 trainable
- Layer4 adapts to surgical domain
- Preserves low-level features (edges, textures)
- Expected to perform better on surgical data""")

pdf.add_page()
pdf.chapter('Part 5: Benchmark Results')

pdf.text("""Published results on Cholec120 dataset for comparison:

- Aksamentov et al. (2017): MAE = 7.7 minutes
- RSDNet (2019): MAE = 8.1 minutes
- TransLocal (2024): MAE = 7.1 minutes

Target: Getting close to 7-8 minutes MAE would be competitive with state-of-the-art published results.""")

import os
os.makedirs('/Users/georgemathios/AI_Surgery_cw2/docs', exist_ok=True)
pdf.output('/Users/georgemathios/AI_Surgery_cw2/docs/01_Implementation_Guide.pdf')
print("Created: docs/01_Implementation_Guide.pdf")


# Create PDF 2: Deep Dive
pdf2 = PDF()
pdf2.set_auto_page_break(auto=True, margin=15)
pdf2.add_page()

pdf2.set_font('Helvetica', 'B', 18)
pdf2.cell(0, 12, 'Deep Dive: Core Concepts', align='C', new_x='LMARGIN', new_y='NEXT')
pdf2.set_font('Helvetica', 'I', 12)
pdf2.cell(0, 8, 'Understanding the Why Behind Each Design Decision', align='C', new_x='LMARGIN', new_y='NEXT')
pdf2.ln(8)

pdf2.chapter('Deep Dive 1: Self-Supervised Learning with Progress & Elapsed Time')

pdf2.section('Elapsed Time vs Progress: The Critical Difference')
pdf2.text("""These are TWO DIFFERENT concepts serving DIFFERENT purposes:

ELAPSED TIME:
- What it is: How long since surgery started (e.g., 20 minutes)
- How we get it: From video timestamp
- Used as: INPUT to the model (helps it know "when" things happen)

PROGRESS:
- What it is: Fraction of surgery complete (0.0 to 1.0)
- How we get it: Calculated as elapsed/total duration
- Used as: TARGET to predict (self-supervised learning signal)""")

pdf2.section('Why Elapsed Time as Input is Crucial')
pdf2.text("""THE PROBLEM: A frame showing "surgeon using grasper" is ambiguous.
- Could be minute 5 (early) --> 35 min remaining
- Could be minute 45 (late) --> 5 min remaining
- Same visual, completely different predictions needed!

THE SOLUTION: Tell the model what time it is!

By concatenating elapsed time with visual features, the model learns time-conditioned patterns:
- "Grasper + elapsed=0.1 (6 min) --> remaining is approximately 35 min"
- "Grasper + elapsed=0.8 (48 min) --> remaining is approximately 3 min"

The visual appearance PLUS temporal context enables accurate prediction.""")

pdf2.section('Why Progress Prediction Helps (Self-Supervised Learning)')
pdf2.text("""What is self-supervised learning?
- Regular supervised: Human annotates labels (expensive, slow, limited)
- Self-supervised: Labels come FREE from the data itself (scalable!)

Progress = elapsed_time / total_duration

This label costs NOTHING - we compute it automatically from timestamps!

By training to predict progress, the model MUST learn:
- Which phase we are in (early phases = low progress)
- How phases typically progress over time
- Visual cues that indicate progression
- What a "normal" surgery timeline looks like

This understanding TRANSFERS directly to time prediction!""")

pdf2.section('The Mathematical Connection')
pdf2.code("""progress = elapsed / total

Rearranging algebraically:
total = elapsed / progress
remaining = total - elapsed
remaining = elapsed * (1 - progress) / progress

EXAMPLE:
- Elapsed = 20 minutes
- Model predicts progress = 0.60 (60% complete)
- Remaining = 20 * (1-0.60) / 0.60 = 13.3 minutes

If the model learns progress well, it has a strong prior for remaining time!""")

pdf2.add_page()
pdf2.chapter('Deep Dive 2: How Different Losses Combine in Backpropagation')

pdf2.section('The Four Losses Explained')
pdf2.text("""1. CrossEntropyLoss (for phase classification)
   - Measures how wrong our phase prediction probability is
   - Confident and correct (P=0.99): loss = 0.01 (very small)
   - Uncertain (P=0.50): loss = 0.69 (medium)
   - Wrong (P=0.01): loss = 4.6 (large penalty!)

2. L1Loss (for time and progress predictions)
   - Simply the absolute difference in predicted vs actual
   - Predicted=12.3 min, True=15.0 min: loss = 2.7
   - Robust to outliers (unlike L2/MSE)
   - Clinically interpretable: "We are off by X minutes on average" """)

pdf2.section('Why Weighted Combination?')
pdf2.code("""total_loss = 0.3 * phase_loss        # Important for understanding workflow
         + 0.2 * phase_time_loss   # Secondary objective
         + 0.3 * surgery_time_loss # Main objective (highest priority)
         + 0.2 * progress_loss     # Self-supervised signal""")

pdf2.text("""Weights serve two purposes:

1. SCALE BALANCING: Different losses have different magnitudes
   - Phase CE: typically 0.5 - 2.0
   - Surgery time L1: typically 5.0 - 20.0
   - Without weights, larger losses would dominate

2. IMPORTANCE: We want to prioritize certain objectives
   - Surgery time gets 0.3 (main goal)
   - Phase classification gets 0.3 (helps everything)
   - Others get 0.2 (supporting roles)""")

pdf2.section('How Gradients Flow Through the Network')
pdf2.text("""During backpropagation, gradients from ALL losses flow through the ENTIRE shared network:

The gradient of total_loss with respect to any weight equals:
0.3 * (gradient from phase_loss) +
0.2 * (gradient from phase_time_loss) +
0.3 * (gradient from surgery_time_loss) +
0.2 * (gradient from progress_loss)

This means:

CNN receives gradients saying:
- "Extract features that help classify phase"
- "Extract features that help predict time"
- "Extract features that help predict progress"

LSTM receives gradients saying:
- "Learn temporal patterns for phase transitions"
- "Learn temporal patterns for time estimation"
- "Learn temporal patterns for progress tracking"

The network learns features useful for ALL tasks simultaneously!""")

pdf2.add_page()
pdf2.chapter('Deep Dive 3: How CNN Features + Elapsed Time Flow Through LSTM')

pdf2.section('Step 1: CNN Extracts Visual Features')
pdf2.text("""Input: Video frame [3, 224, 224] - raw RGB pixels
Output: Feature vector [2048] - semantic description

The CNN (ResNet-50) transforms raw pixels into meaningful numbers that encode:
- Tool presence and position
- Tissue appearance and color
- Anatomical structures visible
- Surgical actions being performed

These 2048 numbers are a compressed "understanding" of the image.""")

pdf2.section('Step 2: Concatenate Elapsed Time')
pdf2.code("""visual_features = CNN(frame)        # [2048] - WHAT we see
elapsed_normalized = [0.52]         # [1] - WHEN we see it (52% of expected duration)
combined = concat(visual, elapsed)  # [2049] - Complete context

This simple concatenation allows the network to learn ANY function
that combines visual and temporal information.""")

pdf2.section('Step 3: LSTM Processes the Sequence')
pdf2.text("""The LSTM receives 30 frames, each with 2049 features.

At each timestep, the LSTM cell:
1. Receives: current input (frame features) + previous memory
2. Decides through learned gates:
   - What to FORGET from old memory
   - What to ADD from new input
   - What to OUTPUT
3. Produces: updated memory + hidden state output

After processing all 30 frames, the final hidden state [256] encodes:
- What tools/anatomy were visible across the 30-second window
- How the scene changed over time
- The elapsed time context at each moment
- Learned patterns that correlate with remaining time""")

pdf2.section('Why This Architecture Enables Accurate Prediction')
pdf2.text("""WITHOUT elapsed time:
- Model sees identical grasper images from two surgeries
- Surgery A: grasper at minute 10, 35 min remaining
- Surgery B: grasper at minute 50, 5 min remaining
- Same input, different outputs needed --> IMPOSSIBLE

WITH elapsed time:
- Model receives: grasper features + elapsed=0.17 --> learns "early, lots remaining"
- Model receives: grasper features + elapsed=0.83 --> learns "late, almost done"
- Different inputs for different situations --> LEARNABLE

The key insight: The LSTM learns to combine:
1. WHAT it sees (from CNN visual features)
2. WHEN it sees it (from elapsed time)
3. HOW things changed (from processing the sequence)

This three-way combination enables accurate, context-aware time prediction!""")

pdf2.output('/Users/georgemathios/AI_Surgery_cw2/docs/02_Deep_Dive_Concepts.pdf')
print("Created: docs/02_Deep_Dive_Concepts.pdf")

print("\nBoth PDFs created successfully in docs/ folder!")

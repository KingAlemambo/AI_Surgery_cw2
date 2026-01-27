"""
Generate PDF explanations for Surgical Duration Prediction project.
"""
from fpdf import FPDF


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font('DejaVu', '', '/System/Library/Fonts/Supplemental/Arial.ttf', uni=True)

    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Surgical Duration Prediction - Deep Learning Guide', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 10, title, 0, 1, 'L', True)
        self.ln(4)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bullet_point(self, text):
        self.set_font('Helvetica', '', 10)
        self.cell(5, 5, chr(149), 0, 0)
        self.multi_cell(0, 5, text)

    def code_block(self, code):
        self.set_font('Courier', '', 9)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(0, 4, code, 0, 'L', True)
        self.ln(2)
        self.set_font('Helvetica', '', 10)


def create_implementation_guide():
    """Create the first PDF - Implementation Guide"""
    pdf = PDF()
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 15, 'Surgical Duration Prediction', 0, 1, 'C')
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Implementation Guide', 0, 1, 'C')
    pdf.ln(10)

    # Part 1: The Problem
    pdf.chapter_title('Part 1: The Problem We Are Solving')

    pdf.section_title('What is the Task?')
    pdf.body_text(
        "Given a video of a laparoscopic cholecystectomy (gallbladder removal surgery), "
        "we want to predict at any moment:\n\n"
        "1. How much longer will the current phase last? (phase remaining time)\n"
        "2. How much longer will the entire surgery last? (surgery remaining time)\n"
        "3. Which phase are we in right now? (phase classification)"
    )

    pdf.section_title('Why Is This Hard?')
    pdf.bullet_point("Surgeries vary wildly: One cholecystectomy might take 25 minutes, another 90 minutes")
    pdf.bullet_point("Same visual appearance, different times: A grasper tool could mean 5 min or 40 min remaining")
    pdf.bullet_point("Single frames are ambiguous: You need to understand the sequence of events")

    pdf.section_title('Clinical Value')
    pdf.body_text(
        "- OR schedulers: Know when to prepare next patient\n"
        "- Surgical team: Anticipate upcoming phases\n"
        "- Anesthesiologists: Plan medication timing\n"
        "- Hospital admin: Optimize resource allocation"
    )

    # Part 2: Architecture
    pdf.add_page()
    pdf.chapter_title('Part 2: Our Architecture - CNN + LSTM')

    pdf.section_title('The Big Picture')
    pdf.body_text(
        "Our model has two main components:\n\n"
        "1. CNN (ResNet-50): Looks at each video frame and extracts visual features - "
        "'what is in this image?'\n\n"
        "2. LSTM: Processes the sequence of features over time - "
        "'given what I have seen so far, what comes next?'"
    )

    pdf.section_title('Data Flow')
    pdf.code_block(
        "Video Frames [B, 30, 3, 224, 224]\n"
        "        |\n"
        "        v\n"
        "    ResNet-50 (CNN) -> extracts 2048-dim features per frame\n"
        "        |\n"
        "        v\n"
        "Features [B, 30, 2048] + Elapsed Time [B, 30, 1]\n"
        "        |\n"
        "        v\n"
        "    LSTM (256 hidden) -> temporal reasoning\n"
        "        |\n"
        "        v\n"
        "    Hidden State [B, 256]\n"
        "        |\n"
        "        v\n"
        "    4 Prediction Heads:\n"
        "    - Phase classification (7 classes)\n"
        "    - Phase time remaining\n"
        "    - Surgery time remaining\n"
        "    - Progress (0-1)"
    )

    pdf.section_title('Why ResNet-50?')
    pdf.bullet_point("Deep enough: 50 layers can learn complex hierarchical features")
    pdf.bullet_point("Residual connections: Solves vanishing gradient problem")
    pdf.bullet_point("Pretrained on ImageNet: Already knows edges, textures, shapes, objects")

    pdf.section_title('Why LSTM?')
    pdf.bullet_point("Temporal dependencies: Surgery has structure - phases follow specific order")
    pdf.bullet_point("Memory: Can remember what happened earlier in the sequence")
    pdf.bullet_point("Context: '10 minutes of dissection' is more informative than a single frame")

    pdf.section_title('Why Multi-Task Learning?')
    pdf.body_text(
        "We train ONE model to predict FOUR things simultaneously. Benefits:\n\n"
        "1. Tasks help each other: Phase recognition helps time prediction\n"
        "2. Shared representation: CNN+LSTM learns features useful for all tasks\n"
        "3. Regularization: Each task constrains learning, preventing overfitting\n"
        "4. Efficiency: One forward pass, multiple outputs"
    )

    # Part 3: Key Papers
    pdf.add_page()
    pdf.chapter_title('Part 3: Key Papers and Their Contributions')

    pdf.section_title('1. EndoNet (Twinanda et al., IEEE-TMI 2016)')
    pdf.body_text(
        "Title: 'EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos'\n\n"
        "Key contributions:\n"
        "- First CNN for multi-task learning on surgical videos\n"
        "- Introduced Cholec80 dataset (80 cholecystectomy videos)\n"
        "- Multi-task: phase recognition + tool detection together\n\n"
        "What we used: Multi-task architecture concept, Cholec80 dataset structure"
    )

    pdf.section_title('2. Aksamentov et al. (MICCAI 2017)')
    pdf.body_text(
        "Title: 'Deep Neural Networks Predict Remaining Surgery Duration from Cholecystectomy Videos'\n\n"
        "Key contributions:\n"
        "- CNN + LSTM architecture for time prediction\n"
        "- Elapsed time as input to LSTM (crucial insight!)\n"
        "- L1 loss for regression (robust to outliers)\n"
        "- Achieved MAE of 7.7 minutes on Cholec120\n\n"
        "What we used: Elapsed time input, L1 loss, CNN-LSTM pipeline"
    )

    pdf.section_title('3. RSDNet (Twinanda et al., IEEE-TMI 2019)')
    pdf.body_text(
        "Title: 'RSDNet: Learning to Predict Remaining Surgery Duration Without Manual Annotations'\n\n"
        "Key contributions:\n"
        "- Progress signal as self-supervised learning (no annotation needed!)\n"
        "- Progress = elapsed_time / total_duration (free from timestamps)\n"
        "- Learning progress helps learn time prediction\n"
        "- Achieved MAE of 8.1 minutes on Cholec120\n\n"
        "What we used: Progress prediction head, self-supervised signal concept"
    )

    pdf.section_title('4. Less is More (Yengera et al., 2018)')
    pdf.body_text(
        "Title: 'Surgical Phase Recognition with Less Annotations through Self-Supervised Pre-training'\n\n"
        "Key contributions:\n"
        "- RSD prediction as pre-training task\n"
        "- End-to-end CNN-LSTM training\n"
        "- Shows how progress understanding transfers to phase recognition\n\n"
        "What we used: Concept of progress as auxiliary task"
    )

    # Part 4: Implementation Details
    pdf.add_page()
    pdf.chapter_title('Part 4: Implementation Details')

    pdf.section_title('Image Preprocessing')
    pdf.body_text(
        "1. Resize to 224x224 (ResNet standard size)\n"
        "2. ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]\n"
        "3. Training augmentation: RandomCrop, HorizontalFlip, ColorJitter\n"
        "4. No augmentation for validation (deterministic evaluation)"
    )

    pdf.section_title('Loss Function')
    pdf.code_block(
        "total_loss = 0.3 * phase_loss        # CrossEntropy for classification\n"
        "           + 0.2 * phase_time_loss   # L1 for regression\n"
        "           + 0.3 * surgery_time_loss # L1 for regression\n"
        "           + 0.2 * progress_loss     # L1 for regression"
    )

    pdf.section_title('Training Configuration')
    pdf.body_text(
        "- Optimizer: Adam with different learning rates\n"
        "  - LSTM and heads: lr = 1e-4 (learning from scratch)\n"
        "  - CNN layer4: lr = 1e-5 (fine-tuning pretrained weights)\n"
        "- Batch size: 4 (limited by GPU memory for video sequences)\n"
        "- Early stopping: patience = 3 epochs\n"
        "- Sequence length: 30 frames (30 seconds at 1fps)"
    )

    pdf.section_title('Freezing Strategy')
    pdf.body_text(
        "Frozen CNN: All ResNet layers frozen, only LSTM trains\n"
        "  - Fast training, uses ImageNet features as-is\n"
        "  - Good baseline, but may not capture surgical-specific features\n\n"
        "Unfrozen Layer4: Early layers frozen, layer4 trainable\n"
        "  - Layer4 adapts to surgical domain\n"
        "  - Preserves low-level features (edges, textures)\n"
        "  - Expected to perform better"
    )

    # Part 5: Expected Results
    pdf.add_page()
    pdf.chapter_title('Part 5: Benchmark Results to Compare Against')

    pdf.body_text(
        "Published results on Cholec120 dataset:\n\n"
        "- Aksamentov et al. (2017): MAE = 7.7 minutes\n"
        "- RSDNet (2019): MAE = 8.1 minutes\n"
        "- TransLocal (2024): MAE = 7.1 minutes\n\n"
        "Target: Getting close to 7-8 minutes MAE would be competitive with published results."
    )

    pdf.output('/Users/georgemathios/AI_Surgery_cw2/docs/01_Implementation_Guide.pdf')
    print("Created: docs/01_Implementation_Guide.pdf")


def create_deep_dive():
    """Create the second PDF - Deep Dive Explanations"""
    pdf = PDF()
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 15, 'Surgical Duration Prediction', 0, 1, 'C')
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Deep Dive: Core Concepts', 0, 1, 'C')
    pdf.ln(10)

    # Deep Dive 1: Self-Supervised Learning
    pdf.chapter_title('Deep Dive 1: Self-Supervised Learning')

    pdf.section_title('Elapsed Time vs Progress: The Difference')
    pdf.body_text(
        "These are TWO DIFFERENT concepts serving DIFFERENT purposes:\n\n"
        "ELAPSED TIME:\n"
        "- What it is: How long since surgery started\n"
        "- How we get it: From video timestamp\n"
        "- Used as: INPUT to the model\n\n"
        "PROGRESS:\n"
        "- What it is: Fraction of surgery complete (0 to 1)\n"
        "- How we get it: Calculated as elapsed/total\n"
        "- Used as: TARGET to predict (self-supervised)"
    )

    pdf.section_title('Why Elapsed Time as Input Helps')
    pdf.body_text(
        "The Problem: A frame showing 'surgeon using grasper' is ambiguous.\n"
        "- Could be minute 5 (early) -> 35 min remaining\n"
        "- Could be minute 45 (late) -> 5 min remaining\n\n"
        "The Solution: Tell the model what time it is!\n\n"
        "With elapsed time, the model can learn time-conditioned patterns:\n"
        "- 'Grasper + elapsed=0.1 (6 min) -> remaining ~ 35 min'\n"
        "- 'Grasper + elapsed=0.8 (48 min) -> remaining ~ 3 min'"
    )

    pdf.section_title('Why Progress Prediction Helps (Self-Supervised)')
    pdf.body_text(
        "What is self-supervised learning?\n"
        "- Regular: Human annotates labels (expensive)\n"
        "- Self-supervised: Labels come FREE from the data itself\n\n"
        "Progress = elapsed / total_duration\n"
        "This label is FREE - we get it from timestamps!\n\n"
        "By training to predict progress, the model must learn:\n"
        "- Which phase we're in (early phases = low progress)\n"
        "- How phases typically progress\n"
        "- Visual cues that indicate progression\n\n"
        "This understanding TRANSFERS to time prediction!"
    )

    pdf.section_title('Mathematical Connection')
    pdf.code_block(
        "progress = elapsed / total\n"
        "\n"
        "Rearranging:\n"
        "total = elapsed / progress\n"
        "remaining = total - elapsed\n"
        "remaining = elapsed * (1 - progress) / progress\n"
        "\n"
        "Example:\n"
        "- Elapsed = 20 minutes\n"
        "- Model predicts progress = 0.60 (60% complete)\n"
        "- Remaining = 20 * (1-0.60) / 0.60 = 13.3 minutes"
    )

    # Deep Dive 2: Loss Combination
    pdf.add_page()
    pdf.chapter_title('Deep Dive 2: How Losses Combine in Backpropagation')

    pdf.section_title('The Four Losses')
    pdf.body_text(
        "1. CrossEntropyLoss (for phase classification)\n"
        "   - Measures how wrong phase prediction is\n"
        "   - Confident and correct (P=0.99): loss = 0.01\n"
        "   - Uncertain (P=0.50): loss = 0.69\n"
        "   - Wrong (P=0.01): loss = 4.6\n\n"
        "2. L1Loss (for time predictions)\n"
        "   - Simply the absolute difference in minutes\n"
        "   - Predicted=12.3, True=15.0: loss = 2.7\n"
        "   - Clinically interpretable!"
    )

    pdf.section_title('Why Weighted Combination?')
    pdf.code_block(
        "total_loss = 0.3 * phase_loss \n"
        "           + 0.2 * phase_time_loss\n"
        "           + 0.3 * surgery_time_loss\n"
        "           + 0.2 * progress_loss"
    )
    pdf.body_text(
        "Weights balance:\n"
        "1. Scale: Different losses have different magnitudes\n"
        "2. Importance: Main objectives (phase, surgery_time) get higher weights"
    )

    pdf.section_title('How Gradients Flow')
    pdf.body_text(
        "During backpropagation, gradients from ALL losses flow through the ENTIRE network:\n\n"
        "CNN receives gradients saying:\n"
        "- 'Extract features that help classify phase'\n"
        "- 'Extract features that help predict time'\n"
        "- 'Extract features that help predict progress'\n\n"
        "LSTM receives gradients saying:\n"
        "- 'Learn temporal patterns for phase transitions'\n"
        "- 'Learn temporal patterns for time estimation'\n"
        "- 'Learn temporal patterns for progress tracking'\n\n"
        "The network learns features useful for ALL tasks, making it more robust!"
    )

    # Deep Dive 3: CNN to LSTM Flow
    pdf.add_page()
    pdf.chapter_title('Deep Dive 3: CNN Features + Elapsed Time through LSTM')

    pdf.section_title('Step 1: CNN Extracts Visual Features')
    pdf.body_text(
        "Input: Video frame [3, 224, 224] - RGB image\n"
        "Output: Feature vector [2048] - semantic description\n\n"
        "Those 2048 numbers encode:\n"
        "- Tool presence and position\n"
        "- Tissue appearance\n"
        "- Anatomical structures\n"
        "- Surgical actions happening"
    )

    pdf.section_title('Step 2: Add Temporal Context')
    pdf.code_block(
        "visual_features = CNN(frame)     # [2048] what we see\n"
        "elapsed_normalized = [0.52]      # [1] when we see it\n"
        "combined = concat(visual, elapsed)  # [2049]"
    )

    pdf.section_title('Step 3: LSTM Processes Sequence')
    pdf.body_text(
        "LSTM receives 30 frames, each with 2049 features.\n\n"
        "At each timestep, LSTM:\n"
        "1. Receives: current input + previous memory\n"
        "2. Decides: what to forget, what to add, what to output\n"
        "3. Produces: updated memory + output\n\n"
        "After 30 frames, the hidden state encodes:\n"
        "- What tools/anatomy were visible across 30 seconds\n"
        "- How the scene changed over time\n"
        "- The elapsed time context at each moment\n"
        "- Patterns that indicate which phase we're in\n"
        "- Signals that correlate with remaining time"
    )

    pdf.section_title('Why This Architecture Works')
    pdf.body_text(
        "Without elapsed time:\n"
        "- Model sees identical grasper images\n"
        "- Must predict different remaining times\n"
        "- IMPOSSIBLE without temporal context!\n\n"
        "With elapsed time:\n"
        "- Model learns: 'Grasper at 40min = near end'\n"
        "- Model learns: 'Grasper at 10min = early stage'\n"
        "- TIME-CONDITIONED visual patterns!"
    )

    pdf.section_title('Key Insight')
    pdf.body_text(
        "The LSTM learns to combine:\n"
        "- WHAT it sees (from CNN features)\n"
        "- WHEN it sees it (from elapsed time)\n"
        "- HOW things changed (from sequence processing)\n\n"
        "This combination enables accurate time prediction!"
    )

    pdf.output('/Users/georgemathios/AI_Surgery_cw2/docs/02_Deep_Dive_Concepts.pdf')
    print("Created: docs/02_Deep_Dive_Concepts.pdf")


if __name__ == "__main__":
    import os
    os.makedirs('/Users/georgemathios/AI_Surgery_cw2/docs', exist_ok=True)

    print("Generating PDFs...")
    create_implementation_guide()
    create_deep_dive()
    print("\nDone! PDFs saved in docs/ folder")

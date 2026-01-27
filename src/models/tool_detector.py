"""
Tool Detection Models for Task B

This module contains models for surgical tool detection:
1. ToolDetectorBaseline - CNN + LSTM with multi-task (tools + phase), no time info
2. ToolDetectorTimed - CNN + LSTM with multi-task + time features from Task A

Following EndoNet (Twinanda et al., 2016):
- Multi-task learning (phase + tools) improves both tasks
- Shared CNN features learn representations useful for both

The Task B experiment:
- Baseline: Multi-task (tools + phase) WITHOUT time
- Timed: Multi-task (tools + phase) WITH time features
- Compare: Does adding estimated times improve tool detection?
"""

import torch
import torch.nn as nn
from models.lstm import TemporalLSTM


class ToolDetectorBaseline(nn.Module):
    """
    Baseline model for surgical tool detection (EndoNet-style multi-task).

    Architecture:
        Video frames → CNN (ResNet-50) → LSTM → Two heads:
                                                 ├─ Tool detection (7 binary)
                                                 └─ Phase classification (7 classes)

    Multi-task learning benefits (from EndoNet):
    - Tools and phases are correlated (e.g., Clipper ↔ ClippingCutting)
    - Shared features learn representations useful for both tasks
    - Acts as regularization

    This baseline uses ONLY visual information, no time features.

    Args:
        cnn: Feature extractor (ResNet50_FeatureExtractor)
        hidden_dim: LSTM hidden dimension (default: 256)
        num_tools: Number of tools to detect (default: 7 for Cholec80)
        num_phases: Number of surgical phases (default: 7 for Cholec80)
        dropout: Dropout probability (default: 0.5)
    """

    def __init__(self, cnn, hidden_dim=256, num_tools=7, num_phases=7, dropout=0.5):
        super().__init__()

        self.cnn = cnn
        self.num_tools = num_tools
        self.num_phases = num_phases
        feature_dim = cnn.feature_dim  # 2048 for ResNet-50

        # Temporal model - processes sequence of CNN features
        self.temporal = TemporalLSTM(
            input_dim=feature_dim,
            hidden_dim=hidden_dim
        )

        # Regularization
        self.dropout = nn.Dropout(p=dropout)

        # Tool detection head (multi-label binary classification)
        # Each tool is independent - use sigmoid, not softmax
        self.tool_head = nn.Linear(hidden_dim, num_tools)

        # Phase classification head (multi-class classification)
        # One phase at a time - use softmax/cross-entropy
        self.phase_head = nn.Linear(hidden_dim, num_phases)

    def forward(self, x):
        """
        Forward pass for multi-task tool + phase prediction.

        Args:
            x: Video frames [B, T, 3, H, W]
               B = batch size, T = sequence length

        Returns:
            dict with:
                - tool_logits: [B, num_tools] - apply sigmoid for probabilities
                - phase_logits: [B, num_phases] - apply softmax for probabilities
        """
        B, T, C, H, W = x.shape

        # Step 1: Extract spatial features from each frame
        feats = self.cnn(x)  # [B, T, 2048]

        # Step 2: Process through LSTM for temporal reasoning
        lstm_out = self.temporal(feats)  # [B, T, hidden_dim]

        # Use last timestep's hidden state (current moment)
        current_state = lstm_out[:, -1, :]  # [B, hidden_dim]

        # Step 3: Apply dropout
        current_state = self.dropout(current_state)

        # Step 4: Multi-task predictions
        tool_logits = self.tool_head(current_state)    # [B, num_tools]
        phase_logits = self.phase_head(current_state)  # [B, num_phases]

        return {
            "tool_logits": tool_logits,
            "phase_logits": phase_logits
        }


class ToolDetectorTimed(nn.Module):
    """
    Timed model for surgical tool detection (Task B main experiment).

    Architecture:
        Video frames → CNN (ResNet-50) ─┐
                                        ├─→ LSTM → Two heads:
        Time features ──────────────────┘          ├─ Tool detection
                                                   └─ Phase classification

    This model adds time features from Task A to the multi-task baseline.

    Hypothesis: Knowing WHERE we are in surgery helps predict WHICH tools.
    - SpecimenBag typically appears near END of surgery
    - Clipper appears during ClippingCutting phase
    - Grasper usage patterns change over time

    Time features can include:
        - elapsed_time: How long since surgery started (normalized)
        - predicted_surgery_remaining: From Task A model
        - predicted_phase_remaining: From Task A model
        - progress: elapsed / total duration

    Args:
        cnn: Feature extractor (ResNet50_FeatureExtractor)
        hidden_dim: LSTM hidden dimension (default: 256)
        num_tools: Number of tools to detect (default: 7)
        num_phases: Number of surgical phases (default: 7)
        num_time_features: Number of time-related inputs (default: 1)
        dropout: Dropout probability (default: 0.5)
    """

    def __init__(self, cnn, hidden_dim=256, num_tools=7, num_phases=7,
                 num_time_features=1, dropout=0.5):
        super().__init__()

        self.cnn = cnn
        self.num_tools = num_tools
        self.num_phases = num_phases
        self.num_time_features = num_time_features
        feature_dim = cnn.feature_dim  # 2048 for ResNet-50

        # Temporal model - processes CNN features + time features
        self.temporal = TemporalLSTM(
            input_dim=feature_dim + num_time_features,
            hidden_dim=hidden_dim
        )

        # Regularization
        self.dropout = nn.Dropout(p=dropout)

        # Multi-task heads
        self.tool_head = nn.Linear(hidden_dim, num_tools)
        self.phase_head = nn.Linear(hidden_dim, num_phases)

    def forward(self, x, time_features):
        """
        Forward pass with time features.

        Args:
            x: Video frames [B, T, 3, H, W]
            time_features: Time information [B, T, num_time_features]
                          e.g., elapsed_time, predicted_remaining, progress

        Returns:
            dict with:
                - tool_logits: [B, num_tools]
                - phase_logits: [B, num_phases]
        """
        B, T, C, H, W = x.shape

        # Step 1: Extract spatial features
        feats = self.cnn(x)  # [B, T, 2048]

        # Step 2: Concatenate time features with visual features
        feats_with_time = torch.cat([feats, time_features], dim=-1)  # [B, T, 2048+K]

        # Step 3: Process through LSTM
        lstm_out = self.temporal(feats_with_time)  # [B, T, hidden_dim]

        # Use last timestep
        current_state = lstm_out[:, -1, :]  # [B, hidden_dim]

        # Step 4: Dropout and multi-task predictions
        current_state = self.dropout(current_state)

        tool_logits = self.tool_head(current_state)    # [B, num_tools]
        phase_logits = self.phase_head(current_state)  # [B, num_phases]

        return {
            "tool_logits": tool_logits,
            "phase_logits": phase_logits
        }


# ============================================================================
# Tool names for Cholec80 dataset (for reference and visualization)
# ============================================================================
CHOLEC80_TOOLS = [
    "Grasper",      # 0 - Most common, used throughout
    "Bipolar",      # 1 - Cauterization
    "Hook",         # 2 - Dissection
    "Scissors",     # 3 - Cutting
    "Clipper",      # 4 - Clipping vessels (specific to ClippingCutting phase)
    "Irrigator",    # 5 - Cleaning
    "SpecimenBag"   # 6 - End of surgery (GallbladderPackaging phase)
]

CHOLEC80_PHASES = [
    "Preparation",              # 0
    "CalotTriangleDissection",  # 1
    "ClippingCutting",          # 2
    "GallbladderDissection",    # 3
    "GallbladderPackaging",     # 4
    "CleaningCoagulation",      # 5
    "GallbladderRetraction"     # 6
]

# Tool-Phase correlations (for understanding, not used in model)
# These correlations are why multi-task learning helps
TOOL_PHASE_CORRELATIONS = {
    "Clipper": ["ClippingCutting"],  # Strong correlation
    "SpecimenBag": ["GallbladderPackaging"],  # Strong correlation
    "Hook": ["CalotTriangleDissection", "GallbladderDissection"],
    "Grasper": ["All phases"],  # Used throughout
}

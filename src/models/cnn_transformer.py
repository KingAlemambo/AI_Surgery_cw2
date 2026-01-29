"""
CNN + Transformer model for surgical workflow analysis (Task A).

Same architecture as CNNLSTMPhaseModel but uses Transformer
instead of LSTM for temporal modeling.
"""

import torch
import torch.nn as nn
from models.transformer import TemporalTransformer


class CNNTransformerPhaseModel(nn.Module):
    """
    CNN + Transformer model for surgical workflow analysis.

    This model combines:
    1. CNN (ResNet-50) for spatial feature extraction
    2. Transformer for temporal modeling
    3. Multiple prediction heads for duration prediction

    Same interface as CNNLSTMPhaseModel for easy comparison.
    """
    def __init__(self,
                 cnn,
                 d_model=256,
                 nhead=8,
                 num_layers=2,
                 num_phases=7,
                 dropout=0.5
    ):
        super().__init__()

        # Step 1: Visual backbone (ResNet-50)
        self.cnn = cnn
        feature_dim = cnn.feature_dim  # 2048 for ResNet-50

        # Step 2: Temporal Model (Transformer instead of LSTM)
        # Input: visual features (2048) + elapsed time (1) = 2049
        self.temporal = TemporalTransformer(
            input_dim=feature_dim + 1,  # +1 for elapsed time
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=0.3  # Dropout inside transformer
        )

        # Dropout for regularization before prediction heads
        self.dropout = nn.Dropout(p=dropout)

        # Step 3: Prediction heads (same as LSTM version)
        # Phase classification head
        self.phase_head = nn.Linear(d_model, num_phases)

        # Time prediction heads (regression)
        self.phase_time_head = nn.Linear(d_model, 1)
        self.surgery_time_head = nn.Linear(d_model, 1)

        # Progress head
        self.progress_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # Task A: Predict start/end times for ALL phases
        self.phase_starts_head = nn.Linear(d_model, num_phases)
        self.phase_ends_head = nn.Linear(d_model, num_phases)

    def forward(self, x, elapsed_time):
        """
        Args:
            x: Video frames [B, T, 3, H, W]
            elapsed_time: Time since surgery start [B, T, 1]

        Returns:
            Dictionary with predictions for each task
        """
        B, T, C, H, W = x.shape

        # Step 1: Extract spatial features from each frame
        feats = self.cnn(x)  # [B, T, 2048]

        # Step 2: Concatenate elapsed time with visual features
        feats_with_time = torch.cat([feats, elapsed_time], dim=-1)  # [B, T, 2049]

        # Step 3: Process through Transformer
        transformer_out = self.temporal(feats_with_time)  # [B, T, d_model]

        # Use the last timestep's representation for predictions
        current_state = transformer_out[:, -1, :]  # [B, d_model]

        # Apply dropout before prediction heads
        current_state = self.dropout(current_state)

        # Step 4: Generate predictions from each head
        phase_logits = self.phase_head(current_state)
        t_phase_pred = self.phase_time_head(current_state).squeeze(-1)
        t_surgery_pred = self.surgery_time_head(current_state).squeeze(-1)
        progress_pred = self.progress_head(current_state).squeeze(-1)

        # Task A: Predict all phase start/end times
        phase_starts_pred = torch.relu(self.phase_starts_head(current_state))
        phase_ends_pred = torch.relu(self.phase_ends_head(current_state))

        return {
            "phase_logits": phase_logits,
            "t_phase_pred": t_phase_pred,
            "t_surgery_pred": t_surgery_pred,
            "progress_pred": progress_pred,
            "phase_starts_pred": phase_starts_pred,
            "phase_ends_pred": phase_ends_pred,
        }

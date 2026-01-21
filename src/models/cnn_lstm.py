import torch
import torch.nn as nn
from models.lstm import TemporalLSTM


class CNNLSTMPhaseModel(nn.Module):
    """
    CNN + LSTM model for surgical workflow analysis.

    This model combines:
    1. CNN (ResNet-50) for spatial feature extraction from video frames
    2. LSTM for temporal modeling across the frame sequence
    3. Multiple prediction heads for different tasks:
       - Phase classification: what surgical phase are we in?
       - Phase time remaining: how long until this phase ends?
       - Surgery time remaining: how long until surgery ends?
       - Progress: what percentage of surgery is complete? (0-1)

    Key improvement: We concatenate elapsed time with visual features
    before feeding to LSTM. This gives the model explicit temporal
    information to reason about "where we are" in the surgery.
    """
    def __init__(self,
                 cnn,
                 hidden_dim=256,
                 num_phases=7
    ):
        super().__init__()

        # Step 1: Visual backbone (ResNet-50)
        self.cnn = cnn
        feature_dim = cnn.feature_dim  # 2048 for ResNet-50

        # Step 2: Temporal Model
        # Input: visual features (2048) + elapsed time (1) = 2049
        self.temporal = TemporalLSTM(
            input_dim=feature_dim + 1,  # +1 for elapsed time
            hidden_dim=hidden_dim
        )

        # Step 3: Prediction heads (multi-task learning)
        # Phase classification head
        self.phase_head = nn.Linear(hidden_dim, num_phases)

        # Time prediction heads (regression)
        self.phase_time_head = nn.Linear(hidden_dim, 1)
        self.surgery_time_head = nn.Linear(hidden_dim, 1)

        # Progress head: predicts 0-1 value (% surgery complete)
        # This is a self-supervised signal - we know progress from timestamps
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Constrain output to [0, 1]
        )

    def forward(self, x, elapsed_time):
        """
        Args:
            x: Video frames [B, T, 3, H, W]
               B = batch size, T = sequence length
            elapsed_time: Time since surgery start [B, T, 1]
               Normalized (divided by expected max duration, e.g., 60 min)

        Returns:
            Dictionary with predictions for each task
        """
        B, T, C, H, W = x.shape

        # Step 1: Extract spatial features from each frame
        # CNN processes each frame independently
        feats = self.cnn(x)  # [B, T, 2048]

        # Step 2: Concatenate elapsed time with visual features
        # This gives LSTM explicit temporal information
        # The model can learn patterns like "at 20 minutes, if we see X,
        # we're likely in phase Y with Z minutes remaining"
        feats_with_time = torch.cat([feats, elapsed_time], dim=-1)  # [B, T, 2049]

        # Step 3: Process through LSTM for temporal reasoning
        lstm_out = self.temporal(feats_with_time)  # [B, T, hidden_dim]

        # Use the last timestep's hidden state for predictions
        # This represents the model's understanding at the current moment
        current_state = lstm_out[:, -1, :]  # [B, hidden_dim]

        # Step 4: Generate predictions from each head
        phase_logits = self.phase_head(current_state)  # [B, num_phases]
        t_phase_pred = self.phase_time_head(current_state).squeeze(-1)  # [B]
        t_surgery_pred = self.surgery_time_head(current_state).squeeze(-1)  # [B]
        progress_pred = self.progress_head(current_state).squeeze(-1)  # [B]

        return {
            "phase_logits": phase_logits,
            "t_phase_pred": t_phase_pred,
            "t_surgery_pred": t_surgery_pred,
            "progress_pred": progress_pred
        }

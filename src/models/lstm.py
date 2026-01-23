import torch.nn as nn


class TemporalLSTM(nn.Module):
    """
    LSTM for temporal modeling of surgical video sequences.

    Why LSTM?
    - Surgery has temporal structure: phases follow specific order
    - Current frame alone isn't enough - context from past frames helps
    - LSTM can learn long-term dependencies (e.g., "we've been in this phase for 5 min")

    How it works:
    - Takes sequence of feature vectors (from CNN)
    - Processes them in order, maintaining hidden state
    - Hidden state accumulates information from past frames
    - Output at each timestep reflects understanding of sequence so far

    Architecture:
    - Input: [B, T, input_dim] - batch of sequences
    - Output: [B, T, hidden_dim] - hidden state at each timestep
    - We typically use the last timestep's output for predictions
    """

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Using 2 layers with dropout between them for regularization
        # Dropout only applies between LSTM layers (not after the last one)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Input shape: [Batch, Time, Features]
            dropout=dropout if num_layers > 1 else 0  # Dropout between layers
        )

    def forward(self, feats_input):
        """
        Process sequence through LSTM.

        Args:
            feats_input: [B, T, input_dim] sequence of feature vectors

        Returns:
            lstm_output: [B, T, hidden_dim] hidden states at each timestep
        """
        # Process through LSTM
        # We ignore the final (h_n, c_n) tuple - we use all timestep outputs
        lstm_output, _ = self.lstm(feats_input)
        return lstm_output

"""
Transformer for temporal modeling of surgical video sequences.

Alternative to LSTM for capturing temporal dependencies.
Transformers use self-attention to model relationships between
all timesteps simultaneously, rather than sequentially like LSTMs.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.

    Since Transformers have no inherent notion of sequence order,
    we add positional information to the input embeddings.
    """
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for temporal modeling.

    Why Transformer?
    - Self-attention captures relationships between ALL timesteps at once
    - No sequential bottleneck like LSTM
    - Can model long-range dependencies more directly
    - Parallelizable during training

    Architecture:
    - Input projection to d_model dimensions
    - Positional encoding
    - N Transformer encoder layers
    - Output: [B, T, d_model]
    """

    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=2, dropout=0.5):
        super().__init__()

        self.d_model = d_model

        # Project input features to transformer dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # Standard: 4x hidden dim
            dropout=dropout,
            batch_first=True  # Input shape: [B, T, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, feats_input):
        """
        Process sequence through Transformer.

        Args:
            feats_input: [B, T, input_dim] sequence of feature vectors

        Returns:
            transformer_output: [B, T, d_model] encoded representations
        """
        # Project to d_model dimensions
        x = self.input_projection(feats_input)  # [B, T, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Process through transformer encoder
        output = self.transformer_encoder(x)  # [B, T, d_model]

        return output

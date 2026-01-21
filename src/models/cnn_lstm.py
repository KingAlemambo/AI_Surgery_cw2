import torch
import torch.nn as nn 
from models.lstm  import TemporalLSTM

class CNNLSTMPhaseModel(nn.Module):
    """
    CNN + LSTM model for surgical phase recognition.
    """
    def __init__(self,
                 cnn,
                 hidden_dim = 256,
                 num_phases=7
    ):
        super().__init__()

        #Step 1 first step the visual backbone model resnet
        self.cnn = cnn
        feature_dim = cnn.feature_dim

        # Step 2 Temporal Model
        self.temporal = TemporalLSTM(
            input_dim= feature_dim,  # 2048
            hidden_dim=hidden_dim
        )

        # step 3 heads
        self.phase_head = nn.Linear(hidden_dim, num_phases)
        self.phase_time_head = nn.Linear(hidden_dim, 1)
        self.surgery_time_head = nn.Linear(hidden_dim, 1)
        #self.classifier = nn.Linear(hidden_dim, num_phases)


    def forward(self, x):
        """
        x: [B, T, 3, H, W]
        """

        # get the spatial features
        feats = self.cnn(x)
        lstm_out = self.temporal(feats)          # [B, T, H]
        current_state = lstm_out[:, -1, :]                 

        # phase prediction
        phase_logits = self.phase_head(current_state)
        t_phase_pred = self.phase_time_head(current_state).squeeze(1)
        t_surgery_pred = self.surgery_time_head(current_state).squeeze(1)

        return {
        "phase_logits": phase_logits,
        "t_phase_pred": t_phase_pred,
        "t_surgery_pred": t_surgery_pred
    }


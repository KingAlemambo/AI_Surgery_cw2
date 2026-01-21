import torch.nn as nn


class TemporalLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size= input_dim,
            hidden_size= hidden_dim,
            num_layers= num_layers,
            batch_first= True
        )

    
    def forward(self, feats_input):

        # we will ignore the final hidden/cell state
        lstm_output, _ = self.lstm(feats_input)
        return lstm_output

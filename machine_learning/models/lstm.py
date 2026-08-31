import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.soc_head = nn.Linear(hidden_size, 1)
        self.soh_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        soc_prediction = self.soc_head(lstm_output)
        pooled_lstm_output = lstm_output.mean(dim=1)
        soh_prediction = self.soh_head(pooled_lstm_output)

        return soc_prediction, soh_prediction

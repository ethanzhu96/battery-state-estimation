"""Static current/voltage SOH baseline."""
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size=2, hidden_layers=2, hidden_dim=64,
                 dropout=0.0, activation='relu'):
        super().__init__()
        activations = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}
        if hidden_layers < 1 or hidden_dim < 1 or not 0 <= dropout < 1 or activation not in activations:
            raise ValueError('Invalid MLP configuration')
        layers = []
        width = input_size
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(width, hidden_dim), activations[activation](), nn.Dropout(dropout)])
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        if features.ndim != 2:
            raise ValueError('MLP expects static [batch, features], not sequences')
        return self.network(features)

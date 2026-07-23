import torch
from torch.utils.data import Dataset

class SOCSequenceDataset(Dataset):
    def __init__(self, features, targets, sequence_length, stride=1):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.sequence_length = sequence_length
        self.stride = stride

        if len(self.features) != len(self.targets):
            raise ValueError("feature length is not equal to target length")

        if sequence_length > len(self.features):
            raise ValueError("sequence length exceeds dataset length")

        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("stride must be > 0")

    def __len__(self):
        return (len(self.features) - self.sequence_length) // self.stride + 1
    
    def __getitem__(self, index):
        start = index * self.stride
        end = start + self.sequence_length

        x_sequence = self.features[start:end]
        y_sequence = self.targets[start:end]

        return x_sequence, y_sequence


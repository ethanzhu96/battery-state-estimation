import torch
from torch.utils.data import Dataset

class SOCSequenceDataset(Dataset):
    def __init__(self, features, targets, sequence_length):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.sequence_length = sequence_length


        if len(self.features) != len(self.targets):
            raise ValueError("feature length is not equal to target length")

        if sequence_length > len(self.features):
            raise ValueError("sequence length exceeds dataset length")

    def __len__(self):
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, index):
        end = index + self.sequence_length

        x_sequence = self.features[index:end]
        y_sequence = self.targets[index:end]

        return x_sequence, y_sequence


from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from machine_learning.datasets.sequence_dataset import SOCSequenceDataset
from machine_learning.models.lstm import LSTM


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("training on", device)

csv_path = Path(__file__).parent / "simulation" / "soc_dataset.csv"
data = pd.read_csv(csv_path)
split_index = int(0.70 * len(data))

train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

train_features = train_data[["current_a", "voltage_v"]].to_numpy()
train_targets = train_data[["soc"]].to_numpy()

test_features = test_data[["current_a", "voltage_v"]].to_numpy()
test_targets = test_data[["soc"]].to_numpy()

# checking the shape :)
# print("Train features:", train_features.shape)
# print("Train targets:", train_targets.shape)
# print("Test features:", test_features.shape)
# print("Test targets:", test_targets.shape)

feature_mean = train_features.mean(axis=0)
feature_std = train_features.std(axis=0)

#pre-processing .. 
train_features_normalized = (train_features - feature_mean) / feature_std
test_features_normalized = (test_features - feature_mean) / feature_std

sequence_length = 300

train_dataset = SOCSequenceDataset(
    features=train_features_normalized,
    targets=train_targets,
    sequence_length=sequence_length,
    stride = 25
)

test_dataset = SOCSequenceDataset(
    features=test_features_normalized,
    targets=test_targets,
    sequence_length=sequence_length,
    stride = 25
)

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)


# batch_features, batch_targets = next(iter(train_loader))

# print("Batch features:", batch_features.shape)
# print("Batch targets:", batch_targets.shape)

model = LSTM(2, 64, 1, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5
loss_fn = nn.MSELoss()

print("Training windows:", len(train_dataset))
print("Test windows:", len(test_dataset))

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch_features, batch_targets in train_loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)

        predictions = model(batch_features)
        loss = loss_fn(predictions, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, loss: {average_loss:.6f}")

model.eval()

total_squared_error = 0.0
total_absolute_error = 0.0
total_values = 0

with torch.no_grad():
    for batch_features, batch_targets in test_loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)

        predictions = model(batch_features)

        squared_errors = (predictions - batch_targets) ** 2
        absolute_errors = torch.abs(predictions - batch_targets)

        total_squared_error += squared_errors.sum().item()
        total_absolute_error += absolute_errors.sum().item()
        total_values += batch_targets.numel()

test_mse = total_squared_error / total_values
test_rmse = test_mse ** 0.5
test_mae = total_absolute_error / total_values

print(f"Test MSE:  {test_mse:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")
print(f"Test MAE:  {test_mae:.6f}")

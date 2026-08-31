import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from machine_learning.datasets.sequence_dataset import SOCSOHSequenceDataset
from machine_learning.models.lstm import LSTM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--soh-only",
        action="store_true",
        help="Train only the SOH objective for the multi-task ablation.",
    )
    return parser.parse_args()


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def trajectory_ids_for_profiles(data, profile_ids):
    trajectory_info = data.groupby("trajectory_id")["profile_id"].first()
    return trajectory_info.index[trajectory_info.isin(profile_ids)].tolist()


def evaluate(model, data_loader, device):
    model.eval()
    soc_squared_error = 0.0
    soc_absolute_error = 0.0
    soc_values = 0
    soh_squared_error = 0.0
    soh_absolute_error = 0.0
    soh_values = 0
    all_soh_targets = []
    all_soh_predictions = []

    with torch.no_grad():
        for features, soc_targets, soh_targets in data_loader:
            features = features.to(device)
            soc_targets = soc_targets.to(device)
            soh_targets = soh_targets.to(device)

            soc_predictions, soh_predictions = model(features)
            soc_errors = soc_predictions - soc_targets
            soh_errors = soh_predictions - soh_targets

            soc_squared_error += (soc_errors ** 2).sum().item()
            soc_absolute_error += soc_errors.abs().sum().item()
            soc_values += soc_targets.numel()
            soh_squared_error += (soh_errors ** 2).sum().item()
            soh_absolute_error += soh_errors.abs().sum().item()
            soh_values += soh_targets.numel()
            all_soh_targets.extend(soh_targets.cpu().flatten().tolist())
            all_soh_predictions.extend(
                soh_predictions.cpu().flatten().tolist()
            )

    return {
        "soc_rmse": (soc_squared_error / soc_values) ** 0.5,
        "soc_mae": soc_absolute_error / soc_values,
        "soh_rmse": (soh_squared_error / soh_values) ** 0.5,
        "soh_mae": soh_absolute_error / soh_values,
        "soh_targets": all_soh_targets,
        "soh_predictions": all_soh_predictions,
    }


def main():
    args = parse_args()
    torch.manual_seed(42)
    device = choose_device()
    print("Training on", device)
    print("Training mode:", "SOH only" if args.soh_only else "SOC + SOH")

    csv_path = Path(__file__).parent / "simulation" / "clean_trajectories.csv"
    data = pd.read_csv(csv_path)

    # Every split contains every SOH and initial SOC. Current profiles are held
    # out so validation and testing measure generalization to unseen loads.
    train_profiles = ["square", "pulse_rest"]
    validation_profiles = ["variable"]
    test_profiles = ["mixed"]
    train_trajectory_ids = trajectory_ids_for_profiles(data, train_profiles)
    validation_trajectory_ids = trajectory_ids_for_profiles(
        data,
        validation_profiles,
    )
    test_trajectory_ids = trajectory_ids_for_profiles(data, test_profiles)

    if not all([
        train_trajectory_ids,
        validation_trajectory_ids,
        test_trajectory_ids,
    ]):
        raise ValueError("One or more current-profile splits are empty")

    feature_columns = ["current_a", "voltage_v"]
    train_rows = data["trajectory_id"].isin(train_trajectory_ids)
    feature_mean = data.loc[train_rows, feature_columns].mean()
    feature_std = data.loc[train_rows, feature_columns].std(ddof=0)
    if (feature_std == 0).any():
        raise ValueError("A training feature has zero standard deviation")

    normalized_data = data.copy()
    normalized_data[feature_columns] = (
        normalized_data[feature_columns] - feature_mean
    ) / feature_std

    sequence_length = 1800
    stride = 50
    batch_size = 64

    train_dataset = SOCSOHSequenceDataset(
        normalized_data,
        train_trajectory_ids,
        sequence_length,
        stride,
    )
    validation_dataset = SOCSOHSequenceDataset(
        normalized_data,
        validation_trajectory_ids,
        sequence_length,
        stride,
    )
    test_dataset = SOCSOHSequenceDataset(
        normalized_data,
        test_trajectory_ids,
        sequence_length,
        stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print("Training profiles:", train_profiles)
    print("Validation profiles:", validation_profiles)
    print("Test profiles:", test_profiles)
    print("Training windows:", len(train_dataset))
    print("Validation windows:", len(validation_dataset))
    print("Test windows:", len(test_dataset))

    model = LSTM(input_size=2, hidden_size=64, num_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_soc_loss = 0.0
        total_soh_loss = 0.0
        total_samples = 0

        for features, soc_targets, soh_targets in train_loader:
            features = features.to(device)
            soc_targets = soc_targets.to(device)
            soh_targets = soh_targets.to(device)

            soc_predictions, soh_predictions = model(features)
            soc_loss = loss_fn(soc_predictions, soc_targets)
            soh_loss = loss_fn(soh_predictions, soh_targets)
            loss = soh_loss if args.soh_only else soc_loss + soh_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = features.shape[0]
            total_soc_loss += soc_loss.item() * batch_size_actual
            total_soh_loss += soh_loss.item() * batch_size_actual
            total_samples += batch_size_actual

        validation_metrics = evaluate(model, validation_loader, device)
        average_soc_loss = total_soc_loss / total_samples
        average_soh_loss = total_soh_loss / total_samples

        if args.soh_only:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"SOH loss: {average_soh_loss:.6f} | "
                f"Validation SOH MAE: "
                f"{100 * validation_metrics['soh_mae']:.3f} percentage points"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"SOC loss: {average_soc_loss:.6f} | "
                f"SOH loss: {average_soh_loss:.6f} | "
                f"Validation SOC RMSE: "
                f"{validation_metrics['soc_rmse']:.6f} | "
                f"Validation SOH MAE: "
                f"{100 * validation_metrics['soh_mae']:.3f} percentage points"
            )

    test_metrics = evaluate(model, test_loader, device)
    print("\nTest results")
    if not args.soh_only:
        print(f"SOC RMSE: {test_metrics['soc_rmse']:.6f}")
        print(f"SOC MAE:  {test_metrics['soc_mae']:.6f}")
    print(
        f"SOH RMSE: {test_metrics['soh_rmse']:.6f} "
        f"({100 * test_metrics['soh_rmse']:.3f} percentage points)"
    )
    print(
        f"SOH MAE:  {test_metrics['soh_mae']:.6f} "
        f"({100 * test_metrics['soh_mae']:.3f} percentage points)"
    )

    soh_results = pd.DataFrame({
        "true_soh": test_metrics["soh_targets"],
        "predicted_soh": test_metrics["soh_predictions"],
    })
    soh_summary = soh_results.groupby("true_soh").agg(
        mean_prediction=("predicted_soh", "mean"),
        prediction_std=("predicted_soh", "std"),
        number_of_windows=("predicted_soh", "size"),
    )
    print("\nSOH predictions by true SOH")
    print(soh_summary.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

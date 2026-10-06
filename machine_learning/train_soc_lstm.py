"""Train a reproducible SOH (or joint SOC/SOH) LSTM experiment."""

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from machine_learning.datasets.sequence_dataset import SOCSOHSequenceDataset
from machine_learning.experiments import (
    apply_preprocessing, create_run_dir, fit_preprocessing, load_checkpoint,
    save_provenance, sha256_file, write_json,
)
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


def build_splits(data, config):
    required = {"trajectory_id", "time_s", "soh", *config["feature_columns"]}
    if config["task"] == "joint":
        required.add("soc")
    if not config["splits_file"]:
        required.add("profile_id")
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Dataset contains missing required values")
    numeric = ["time_s", "soh", *config["feature_columns"]]
    if config["task"] == "joint":
        numeric.append("soc")
    if not np.isfinite(data[numeric].to_numpy()).all():
        raise ValueError("Dataset contains non-finite input/target values")
    for trajectory_id, trajectory in data.groupby("trajectory_id"):
        if trajectory["time_s"].duplicated().any():
            raise ValueError(f"Trajectory {trajectory_id} has duplicate timestamps")
        for column in ("soh", "profile_id", "cell_id", "cycle_id"):
            if column in trajectory and trajectory[column].nunique(dropna=False) != 1:
                raise ValueError(f"Trajectory {trajectory_id} contains multiple {column} values")

    names = ("train", "validation", "test")
    if config["splits_file"]:
        saved = json.loads(Path(config["splits_file"]).read_text(encoding="utf-8"))
        splits = {name: saved[name] for name in names}
        strategy = saved.get("strategy", "explicit_trajectory_ids")
    else:
        profiles = [config[f"{name}_profiles"] for name in names]
        if len(set(sum(profiles, []))) != sum(map(len, profiles)):
            raise ValueError("Training, validation, and test profiles must be disjoint")
        info = data.groupby("trajectory_id")["profile_id"].first()
        missing_profiles = set(sum(profiles, [])) - set(info)
        if missing_profiles:
            raise ValueError(f"Profiles not found in dataset: {sorted(missing_profiles)}")
        splits = {name: info.index[info.isin(config[f"{name}_profiles"])].tolist() for name in names}
        strategy = "profile_holdout"

    used = set()
    available = set(data["trajectory_id"])
    for name, ids in splits.items():
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{name} split must have nonempty, unique trajectory IDs")
        if used.intersection(ids):
            raise ValueError("Trajectory IDs overlap between splits")
        if set(ids) - available:
            raise ValueError(f"{name} split includes IDs absent from the dataset")
        lengths = data[data["trajectory_id"].isin(ids)].groupby("trajectory_id").size()
        if (lengths < config["sequence_length"]).any():
            raise ValueError(f"{name} split has trajectories shorter than sequence_length")
        used.update(ids)
    splits["strategy"] = strategy
    splits["unused_trajectory_ids"] = data.loc[~data["trajectory_id"].isin(used), "trajectory_id"].unique().tolist()
    return splits


def evaluate(model, data_loader, device, task="joint"):
    model.eval()
    sums = {name: [0.0, 0.0, 0.0, 0] for name in ("soh", "soc")}
    predictions, targets = [], []
    with torch.no_grad():
        for batch in data_loader:
            features = batch[0].to(device)
            soh_targets = batch[-1].to(device)
            if task == "joint":
                soc_predictions, soh_predictions = model(features, task=task)
                errors = {"soc": soc_predictions - batch[1].to(device)}
            else:
                soh_predictions = model(features, task=task)
                errors = {}
            errors["soh"] = soh_predictions - soh_targets
            for name, error in errors.items():
                sums[name][0] += error.square().sum().item()
                sums[name][1] += error.abs().sum().item()
                sums[name][2] += error.sum().item()
                sums[name][3] += error.numel()
            predictions.extend(soh_predictions.cpu().flatten().tolist())
            targets.extend(soh_targets.cpu().flatten().tolist())
    metrics = {"soh_predictions": predictions, "soh_targets": targets}
    for name, (squared, absolute, signed, count) in sums.items():
        if count:
            metrics.update({
                f"{name}_rmse": math.sqrt(squared / count),
                f"{name}_mae": absolute / count, f"{name}_bias": signed / count,
            })
    return metrics


def metrics_in_pp(metrics):
    return {f"{key}_pp": value * 100 for key, value in metrics.items()
            if key.endswith(("_rmse", "_mae", "_bias"))}


def train_epoch(model, loader, optimizer, device, task):
    model.train()
    soh_loss_sum = soc_loss_sum = count = 0
    for batch in loader:
        features, soh_targets = batch[0].to(device), batch[-1].to(device)
        if task == "joint":
            soc_predictions, soh_predictions = model(features, task=task)
            soc_loss = nn.functional.mse_loss(soc_predictions, batch[1].to(device))
        else:
            soh_predictions = model(features, task=task)
            soc_loss = torch.zeros((), device=device)
        soh_loss = nn.functional.mse_loss(soh_predictions, soh_targets)
        loss = soh_loss + soc_loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        size = features.shape[0]
        soh_loss_sum += soh_loss.item() * size
        soc_loss_sum += soc_loss.item() * size
        count += size
    metrics = {"train_soh_mse": soh_loss_sum / count,
               "train_soh_rmse_pp": 100 * math.sqrt(soh_loss_sum / count)}
    if task == "joint":
        metrics.update({"train_soc_mse": soc_loss_sum / count,
                        "train_soc_rmse_pp": 100 * math.sqrt(soc_loss_sum / count)})
    return metrics


def run_experiment(config):
    config = validate_config(config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = choose_device(config["device"])
    checksum = sha256_file(config["data_path"])
    if config["data_sha256"] is not None and checksum != config["data_sha256"]:
        raise ValueError("Dataset checksum differs from the saved config; use a new config for changed data")
    config["data_sha256"] = checksum
    data = pd.read_csv(config["data_path"])
    splits = build_splits(data, config)
    preprocessing = fit_preprocessing(
        data[data["trajectory_id"].isin(splits["train"])],
        config["feature_columns"], config["normalization"],
    )
    normalized_data = apply_preprocessing(data, preprocessing)
    datasets = {
        name: SOCSOHSequenceDataset(
            normalized_data, splits[name], config["sequence_length"], config["stride"],
            include_soc=config["task"] == "joint", feature_columns=config["feature_columns"],
        ) for name in ("train", "validation", "test")
    }
    shuffle_generator = torch.Generator().manual_seed(config["seed"])
    loaders = {name: DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=name == "train",
        generator=shuffle_generator if name == "train" else None,
    ) for name, dataset in datasets.items()}

    run_dir = create_run_dir(config["output_dir"], config["run_name"])
    print(f"Run artifacts: {run_dir}", flush=True)
    write_json(run_dir / "status.json", {"status": "running"})
    try:
        # Saved configs replay the exact split and choose a fresh directory automatically.
        config["run_name"] = None
        config["splits_file"] = str(run_dir / "splits.json")
        write_json(run_dir / "config.json", config)
        splits["window_counts"] = {name: len(dataset) for name, dataset in datasets.items()}
        write_json(run_dir / "splits.json", splits)
        write_json(run_dir / "preprocessing.json", preprocessing)
        save_provenance(run_dir, config, device)
        print(f"Training {config['task']} on {device}; windows: {splits['window_counts']}", flush=True)
        model = LSTM(input_size=len(config["feature_columns"]),
                     hidden_size=config["hidden_size"], num_layers=config["num_layers"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        best_rmse, best_epoch, stale_epochs = float("inf"), 0, 0
        with (run_dir / "learning_curves.csv").open("w", newline="", encoding="utf-8") as log:
            writer = None
            for epoch in range(1, config["epochs"] + 1):
                training = train_epoch(model, loaders["train"], optimizer, device, config["task"])
                validation = evaluate(model, loaders["validation"], device, config["task"])
                if not math.isfinite(validation["soh_rmse"]):
                    raise ValueError("Non-finite validation SOH RMSE")
                row = {"epoch": epoch, **training,
                       **{f"val_{key}": value for key, value in metrics_in_pp(validation).items()}}
                if writer is None:
                    writer = csv.DictWriter(log, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                log.flush()
                if validation["soh_rmse"] < best_rmse:
                    best_rmse, best_epoch, stale_epochs = validation["soh_rmse"], epoch, 0
                    torch.save({
                        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "config": config, "preprocessing": preprocessing,
                        "epoch": epoch, "validation_soh_rmse_pp": best_rmse * 100,
                    }, run_dir / "best_model.pt")
                else:
                    stale_epochs += 1
                print(f"Epoch {epoch}/{config['epochs']} | train SOH RMSE {training['train_soh_rmse_pp']:.3f} pp"
                      f" | validation SOH RMSE {100 * validation['soh_rmse']:.3f} pp", flush=True)
                if config["patience"] and stale_epochs >= config["patience"]:
                    print(f"Early stopping; best epoch: {best_epoch}", flush=True)
                    break

        model, _ = load_checkpoint(run_dir / "best_model.pt", device)
        test = evaluate(model, loaders["test"], device, config["task"])
        predictions = pd.DataFrame(datasets["test"].window_metadata)
        predictions["true_soh"] = test["soh_targets"]
        predictions["predicted_soh"] = test["soh_predictions"]
        predictions["error_pp"] = 100 * (predictions["predicted_soh"] - predictions["true_soh"])
        predictions.to_csv(run_dir / "predictions.csv", index=False)
        metrics = {
            "best_epoch": best_epoch, "epochs_completed": epoch,
            "selection_metric": "validation_soh_rmse_pp",
            "best_validation_soh_rmse_pp": best_rmse * 100,
            "test": metrics_in_pp(test),
            "aggregation": "window-weighted; SOC metrics include all timesteps in overlapping windows",
            "target_units": "fraction (0.85 = 85% SOH)",
            "error_units": "percentage points; bias = prediction minus target",
            "test_window_count": len(datasets["test"]),
        }
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "status.json", {"status": "completed"})
        print(f"Test SOH RMSE: {100 * test['soh_rmse']:.3f} pp | MAE: {100 * test['soh_mae']:.3f} pp", flush=True)
    except BaseException as exc:
        write_json(run_dir / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    return run_dir


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

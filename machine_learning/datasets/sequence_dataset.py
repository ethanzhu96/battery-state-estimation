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


class SOCSOHSequenceDataset(Dataset):
    """Create SOC/SOH sequence windows without crossing trajectories."""

    def __init__(self, dataframe, trajectory_ids, sequence_length, stride=1):
        required_columns = {
            "trajectory_id",
            "time_s",
            "current_a",
            "voltage_v",
            "soc",
            "soh",
        }
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"dataframe is missing required columns: {missing}")

        if not isinstance(sequence_length, int) or sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer")

        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("stride must be a positive integer")

        selected_data = dataframe[
            dataframe["trajectory_id"].isin(trajectory_ids)
        ]
        if selected_data.empty:
            raise ValueError("no rows found for the requested trajectory IDs")

        self.sequence_length = sequence_length
        self.trajectories = {}
        self.windows = []

        for trajectory_id, trajectory in selected_data.groupby("trajectory_id"):
            trajectory = trajectory.sort_values("time_s")
            if len(trajectory) < sequence_length:
                continue

            features = torch.tensor(
                trajectory[["current_a", "voltage_v"]].to_numpy(),
                dtype=torch.float32,
            )
            soc_targets = torch.tensor(
                trajectory[["soc"]].to_numpy(),
                dtype=torch.float32,
            )

            soh_values = trajectory["soh"].unique()
            if len(soh_values) != 1:
                raise ValueError(
                    f"trajectory {trajectory_id} contains multiple SOH values"
                )
            soh_target = torch.tensor([soh_values[0]], dtype=torch.float32)

            self.trajectories[trajectory_id] = (
                features,
                soc_targets,
                soh_target,
            )

            number_of_windows = (
                len(trajectory) - sequence_length
            ) // stride + 1
            for window_index in range(number_of_windows):
                start = window_index * stride
                self.windows.append((trajectory_id, start))

        if not self.windows:
            raise ValueError("selected trajectories are shorter than sequence_length")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        trajectory_id, start = self.windows[index]
        end = start + self.sequence_length
        features, soc_targets, soh_target = self.trajectories[trajectory_id]

        x_sequence = features[start:end]
        soc_sequence = soc_targets[start:end]

        return x_sequence, soc_sequence, soh_target

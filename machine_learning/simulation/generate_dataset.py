from pathlib import Path

import pandas as pd
import numpy as np

from current_profile import (
    CLEAN_CURRENT_PROFILES,
    generate_clean_current_profile,
    generate_current_noise,
)
from ecm_2rc import ocv_from_soc, terminal_voltage, update_states


def calculate_aged_parameters(soh, bol_capacity_ah, bol_r0, bol_r1, bol_r2):
    """Apply simple synthetic capacity fade and resistance growth laws."""
    if not 0.0 < soh <= 1.0:
        raise ValueError("soh must be greater than 0 and at most 1")

    degradation = 1.0 - soh
    return {
        "capacity_ah": bol_capacity_ah * soh,
        "r0": bol_r0 * (1.0 + 2.0 * degradation),
        "r1": bol_r1 * (1.0 + 1.5 * degradation),
        "r2": bol_r2 * (1.0 + 1.0 * degradation),
    }


def generate_dataset(
    trajectory_id,
    profile_id,
    soh,
    num_steps,
    dt_s,
    capacity_ah,
    current_magnitude_a,
    r0,
    r1,
    c1,
    r2,
    c2,
    initial_soc,
    noise_std_c,
    seed,
):
    soc_state = initial_soc
    v1_state = 0.0
    v2_state = 0.0
    time_data = np.zeros(num_steps)
    current_data = np.zeros(num_steps)
    voltage_data = np.zeros(num_steps)
    soc_data = np.zeros(num_steps)

    current_noise = generate_current_noise(
        num_steps=num_steps,
        capacity_ah=capacity_ah,
        noise_std_c=noise_std_c,
        seed=seed,
    )
    base_current = generate_clean_current_profile(
        profile_name=profile_id,
        num_steps=num_steps,
        dt_s=dt_s,
        current_magnitude_a=current_magnitude_a,
    )

    for k in range(num_steps):
        time_data[k] = k * dt_s

        current = base_current[k] + current_noise[k]

        ocv = ocv_from_soc(soc_state)

        voltage = terminal_voltage(
            ocv=ocv,
            current_a=current,
            r0=r0,
            v1=v1_state,
            v2=v2_state,
        )

        current_data[k] = current
        voltage_data[k] = voltage
        soc_data[k] = soc_state

        # advance the model to the next time step
        soc_state, v1_state, v2_state = update_states(
            soc=soc_state,
            v1=v1_state,
            v2=v2_state,
            current_a=current,
            dt_s=dt_s,
            capacity_ah=capacity_ah,
            r1=r1,
            c1=c1,
            r2=r2,
            c2=c2,
        )

    dataset = pd.DataFrame({
        "trajectory_id": trajectory_id,
        "profile_id": profile_id,
        "time_s": time_data,
        "current_a": current_data,
        "voltage_v": voltage_data,
        "soc": soc_data,
        "soh": soh,
        "initial_soc": initial_soc,
        "capacity_ah": capacity_ah,
        "r0_ohm": r0,
        "r1_ohm": r1,
        "r2_ohm": r2,
    })

    return dataset

def main():
    bol_capacity_ah = 2.5
    bol_r0 = 0.01
    bol_r1 = 0.015
    bol_r2 = 0.02
    soh_values = [1.0, 0.9, 0.8, 0.7]
    initial_soc_values = [0.2, 0.5, 0.8]
    profile_ids = list(CLEAN_CURRENT_PROFILES)

    dt_s = 1.0
    duration_s = 2 * 3600
    num_steps = int(duration_s / dt_s)

    trajectories = []
    trajectory_id = 0

    for soh in soh_values:
        aged_parameters = calculate_aged_parameters(
            soh=soh,
            bol_capacity_ah=bol_capacity_ah,
            bol_r0=bol_r0,
            bol_r1=bol_r1,
            bol_r2=bol_r2,
        )
        for initial_soc in initial_soc_values:
            for profile_id in profile_ids:
                trajectory = generate_dataset(
                    trajectory_id=trajectory_id,
                    profile_id=profile_id,
                    soh=soh,
                    num_steps=num_steps,
                    dt_s=dt_s,
                    capacity_ah=aged_parameters["capacity_ah"],
                    current_magnitude_a=bol_capacity_ah,
                    r0=aged_parameters["r0"],
                    r1=aged_parameters["r1"],
                    c1=2400.0,
                    r2=aged_parameters["r2"],
                    c2=12000.0,
                    initial_soc=initial_soc,
                    noise_std_c=0.0,
                    seed=trajectory_id,
                )
                trajectories.append(trajectory)
                trajectory_id += 1

    dataset = pd.concat(trajectories, ignore_index=True)

    print(dataset.head())
    print(f"Dataset shape: {dataset.shape}")
    print(f"Number of trajectories: {dataset['trajectory_id'].nunique()}")
    print("\nTrajectory summary:")
    print(dataset.groupby("trajectory_id").agg(
        soh=("soh", "first"),
        initial_soc=("initial_soc", "first"),
        profile_id=("profile_id", "first"),
        capacity_ah=("capacity_ah", "first"),
        r0_ohm=("r0_ohm", "first"),
        r1_ohm=("r1_ohm", "first"),
        r2_ohm=("r2_ohm", "first"),
        min_soc=("soc", "min"),
        max_soc=("soc", "max"),
        min_voltage_v=("voltage_v", "min"),
        max_voltage_v=("voltage_v", "max"),
        num_rows=("time_s", "size"),
    ))
    print(f"Missing values:\n{dataset.isna().sum()}")

    output_path = Path(__file__).with_name("clean_trajectories.csv")
    dataset.to_csv(output_path, index=False)

    print(f"Saved dataset to: {output_path}")


if __name__ == "__main__":
    main()

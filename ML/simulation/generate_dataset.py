from pathlib import Path

import pandas as pd
import numpy as np

from current_profile import switch_mode, generate_current_noise
from ecm_2rc import ocv_from_soc, terminal_voltage, update_states

def generate_dataset(
    num_steps,
    dt_s,
    capacity_ah,
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
    direction = 1

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

    for k in range(num_steps):
        time_data[k] = k * dt_s

        #make sure soc is going the right direction
        base_current, direction = switch_mode(
            soc=soc_state,
            direction=direction,
            capacity_ah=capacity_ah,
        )

        current = base_current + current_noise[k]

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
        "time_s": time_data,
        "current_a": current_data,
        "voltage_v": voltage_data,
        "soc": soc_data,
    })

    return dataset

def main():
    dt_s = 1.0
    duration_s = 20 * 3600
    num_steps = int(duration_s / dt_s)

    dataset = generate_dataset(
        num_steps=num_steps,
        dt_s=dt_s,
        capacity_ah=2.5,
        r0=0.01,
        r1=0.015,
        c1=2400.0,
        r2=0.02,
        c2=12000.0,
        initial_soc=0.1,
        noise_std_c=0.5,
        seed=42,
    )

    print(dataset.head())
    print(f"Dataset shape: {dataset.shape}")
    print(f"SOC range: {dataset['soc'].min():.4f} to "
          f"{dataset['soc'].max():.4f}")
    print(f"Voltage range: {dataset['voltage_v'].min():.4f} to "
          f"{dataset['voltage_v'].max():.4f}")
    print(f"Missing values:\n{dataset.isna().sum()}")

    output_path = Path(__file__).with_name("soc_dataset.csv")
    dataset.to_csv(output_path, index=False)

    print(f"Saved dataset to: {output_path}")


if __name__ == "__main__":
    main()
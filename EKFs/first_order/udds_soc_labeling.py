import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "first_order"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "first_order"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COULOMBIC_EFFICIENCY = 0.995

CSV_CANDIDATES = [
    DATA_DIR / "SAMSUNG_Cell_5_udds_Channel_6_Wb_1.CSV",
    DATA_DIR / "SAMSUNG_Cell_5_udds_Channel_6_Wb_1 copy.CSV",
    MODULE_DIR / "SAMSUNG_Cell_5_udds_Channel_6_Wb_1.CSV",
]

def find_first_existing(candidates, label):
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing {label}. Tried:\n{tried}")

REST_CURRENT_THRESHOLD = 0.01      # A
FULL_CHARGE_VOLTAGE = 4.15         # V, used to detect full-charge region
ACTIVE_DISCHARGE_THRESHOLD = -0.2  # A, start of drive/discharge
TAIL_CURRENT_THRESHOLD = 0.05      # A, end-of-charge taper current

MANUAL_START_IDX = None
MANUAL_END_IDX = None

def clean_columns(df):
    df = df.copy()
    df.columns = [c.replace("ÿ", "").replace("ï»¿", "").strip() for c in df.columns]
    return df

def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding="latin1")
    df = clean_columns(df)

    needed = [
        "Test Time (s)",
        "Current (A)",
        "Voltage (V)",
        "Charge Capacity (Ah)",
        "Discharge Capacity (Ah)",
    ]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=["Test Time (s)", "Current (A)", "Voltage (V)"]).reset_index(drop=True)

    t = df["Test Time (s)"].to_numpy(dtype=float)
    i = df["Current (A)"].to_numpy(dtype=float)
    v = df["Voltage (V)"].to_numpy(dtype=float)
    ccap = np.nan_to_num(df["Charge Capacity (Ah)"].to_numpy(dtype=float), nan=0.0)
    dcap = np.nan_to_num(df["Discharge Capacity (Ah)"].to_numpy(dtype=float), nan=0.0)

    return df, t, i, v, ccap, dcap

def find_first_full_charge_end(t, i, v):
    """
    Find the end of the INITIAL full-charge rest region near the beginning
    of the file, right before the UDDS discharge starts.

    We do NOT want the last high-rest point in the entire file, because the
    file ends at full charge too.
    """
    high_rest = (np.abs(i) <= REST_CURRENT_THRESHOLD) & (v >= FULL_CHARGE_VOLTAGE)

    idx = np.where(high_rest)[0]
    if len(idx) == 0:
        raise ValueError("Could not find initial high-voltage rest region.")

    start_block = [idx[0]]
    for k in idx[1:]:
        if k == start_block[-1] + 1:
            start_block.append(k)
        else:
            break

    last_initial_high_rest = start_block[-1]

    discharge_candidates = np.where(
        (np.arange(len(i)) > last_initial_high_rest) & (i <= ACTIVE_DISCHARGE_THRESHOLD)
    )[0]

    if len(discharge_candidates) == 0:
        raise ValueError("Could not find discharge start after initial full-charge region.")

    first_discharge_idx = discharge_candidates[0]
    start_idx = max(first_discharge_idx - 1, 0)
    return start_idx

def find_second_full_charge_end(t, i, v):
    """
    Find the final full-charge point near the end of the file.
    If the end of the file is a full-charge rest region, use the last point.
    Otherwise fall back to the last row.
    """
    high_rest = (np.abs(i) <= TAIL_CURRENT_THRESHOLD) & (v >= FULL_CHARGE_VOLTAGE)

    idx = np.where(high_rest)[0]
    if len(idx) == 0:
        return len(t) - 1

    end_block = [idx[-1]]
    for k in idx[-2::-1]:
        if k == end_block[-1] - 1:
            end_block.append(k)
        else:
            break

    end_block = sorted(end_block)
    return end_block[-1]

def compute_capacities_and_efficiency(t, i, start_idx, end_idx):
    """
    Between the two full charges:
      Qchg = integral of charging current (I > 0)
      Qdis = integral of discharge magnitude (I < 0)

    Shida defines:
      eta = Qdis / Qchg
    """
    Qchg_as = 0.0
    Qdis_as = 0.0

    for k in range(start_idx, end_idx):
        dt = max(t[k + 1] - t[k], 0.0)
        ik = i[k]

        if ik > 0:
            Qchg_as += ik * dt
        elif ik < 0:
            Qdis_as += (-ik) * dt

    if Qchg_as <= 0:
        raise ValueError("Computed charged capacity is zero. Check charge/discharge sign convention.")
    if Qdis_as <= 0:
        raise ValueError("Computed discharged capacity is zero. Check chosen segment.")

    eta = COULOMBIC_EFFICIENCY
    return Qchg_as, Qdis_as, eta

def coulomb_count_soc(t, i, start_idx, end_idx, eta, Qd_as):
    """
    Shida's rule:
      if I > 0:
          SOC[k+1] = SOC[k] + eta * I * dt / Qd
      if I < 0:
          SOC[k+1] = SOC[k] + I * dt / Qd

    Here:
      positive current = charging
      negative current = discharging

    Start at 100% SOC at the first full charge.
    """
    n = end_idx - start_idx + 1
    soc = np.zeros(n, dtype=float)
    soc[0] = 1.0

    for kk in range(n - 1):
        k = start_idx + kk
        dt = max(t[k + 1] - t[k], 0.0)
        ik = i[k]

        if ik > 0:
            soc[kk + 1] = soc[kk] + eta * ik * dt / Qd_as
        else:
            soc[kk + 1] = soc[kk] + ik * dt / Qd_as

        soc[kk + 1] = np.clip(soc[kk + 1], 0.0, 1.0)

    return soc

def main():
    csv_path = find_first_existing(CSV_CANDIDATES, "raw UDDS CSV")

    df, t, i, v, ccap, dcap = load_data(csv_path)

    if MANUAL_START_IDX is not None:
        start_idx = MANUAL_START_IDX
    else:
        start_idx = find_first_full_charge_end(t, i, v)

    if MANUAL_END_IDX is not None:
        end_idx = MANUAL_END_IDX
    else:
        end_idx = find_second_full_charge_end(t, i, v)

    if end_idx <= start_idx:
        raise ValueError("End index must be after start index.")

    print(f"Segment start index: {start_idx}, time = {t[start_idx]:.3f} s")
    print(f"Segment end index:   {end_idx}, time = {t[end_idx]:.3f} s")

    Qchg_as, Qdis_as, eta = compute_capacities_and_efficiency(t, i, start_idx, end_idx)

    print(f"\nTotal charged capacity    Qchg = {Qchg_as:.6f} A*s  ({Qchg_as/3600:.6f} Ah)")
    print(f"Total discharged capacity Qdis = {Qdis_as:.6f} A*s  ({Qdis_as/3600:.6f} Ah)")
    print(f"Coulombic efficiency eta  = {eta:.8f}")

    Qd_as = Qdis_as
    print(f"Maximum discharge capacity Qd = {Qd_as:.6f} A*s  ({Qd_as/3600:.6f} Ah)")

    soc = coulomb_count_soc(t, i, start_idx, end_idx, eta, Qd_as)

    print(f"\nInitial SOC = {soc[0]:.6f}")
    print(f"Final SOC   = {soc[-1]:.6f}")

    out_df = df.iloc[start_idx:end_idx + 1].copy().reset_index(drop=True)
    out_df["SoC"] = soc
    out_df["SOC_label"] = soc

    out_csv = OUTPUT_DIR / "udds_between_full_charges_with_soc.csv"
    out_df.to_csv(out_csv, index=False)

    summary_path = OUTPUT_DIR / "udds_soc_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Input file: {csv_path.name}\n")
        f.write(f"Segment start index: {start_idx}\n")
        f.write(f"Segment end index: {end_idx}\n")
        f.write(f"Segment start time (s): {t[start_idx]:.6f}\n")
        f.write(f"Segment end time (s): {t[end_idx]:.6f}\n\n")
        f.write(f"Qchg (A*s): {Qchg_as:.10f}\n")
        f.write(f"Qchg (Ah): {Qchg_as/3600:.10f}\n")
        f.write(f"Qdis (A*s): {Qdis_as:.10f}\n")
        f.write(f"Qdis (Ah): {Qdis_as/3600:.10f}\n")
        f.write(f"eta: {eta:.10f}\n")
        f.write(f"Qd (A*s): {Qd_as:.10f}\n")
        f.write(f"Qd (Ah): {Qd_as/3600:.10f}\n")
        f.write(f"Initial SOC: {soc[0]:.10f}\n")
        f.write(f"Final SOC: {soc[-1]:.10f}\n")

    print(f"\nSaved: {out_csv.name}")
    print(f"Saved: {summary_path.name}")

    tt = t[start_idx:end_idx + 1] - t[start_idx]
    ii = i[start_idx:end_idx + 1]
    vv = v[start_idx:end_idx + 1]

    plt.figure(figsize=(12, 4))
    plt.plot(tt, ii)
    plt.xlabel("Time since first full charge (s)")
    plt.ylabel("Current (A)")
    plt.title("UDDS segment current")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "udds_segment_current.png", dpi=220)

    plt.figure(figsize=(12, 4))
    plt.plot(tt, vv)
    plt.xlabel("Time since first full charge (s)")
    plt.ylabel("Voltage (V)")
    plt.title("UDDS segment voltage")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "udds_segment_voltage.png", dpi=220)

    plt.figure(figsize=(12, 4))
    plt.plot(tt, soc)
    plt.xlabel("Time since first full charge (s)")
    plt.ylabel("SOC")
    plt.title("SOC label from Coulomb counting")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "udds_soc_label.png", dpi=220)

    plt.close("all")

if __name__ == "__main__":
    main()

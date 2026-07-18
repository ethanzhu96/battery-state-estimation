import numpy as np
import pandas as pd
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "second_order"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "second_order"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_CANDIDATES = [
    "SAMSUNG_Cell_5_udds_Channel_6_Wb_1.CSV",
    "SAMSUNG_Cell_5_udds_Channel_6_Wb_1 copy.CSV",
    "SAMSUNG_Cell_5_UDDS_Channel_6_Wb_1.CSV",
    "SAMSUNG_Cell_5_UDDS_Channel_6_Wb_1 copy.CSV",
]

REST_CURRENT_THRESHOLD = 0.005
ACTIVE_CURRENT_THRESHOLD = 0.05
MIN_FULL_REST_DURATION_S = 600.0   # 10 min
FULL_VOLTAGE_WINDOW_V = 0.03       # within 30 mV of max rest voltage
EDGE_BUFFER_S = 300.0              # include 5 min around full-charge rests

def find_existing_file(candidates):
    for name in candidates:
        for base in (DATA_DIR, MODULE_DIR):
            path = base / name
            if path.exists():
                return path

    glob_patterns = [
        "*udds*Channel_6*Wb_1*.CSV",
        "*UDDS*Channel_6*Wb_1*.CSV",
        "*udds*.CSV",
        "*UDDS*.CSV",
    ]
    for pattern in glob_patterns:
        matches = sorted(DATA_DIR.glob(pattern)) or sorted(MODULE_DIR.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "Could not find a UDDS CSV. Put the raw UDDS file in data/second_order/ or rename it to one of: "
        + ", ".join(CSV_CANDIDATES)
    )

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).replace("ÿ", "").replace("ï»¿", "").strip() for c in df.columns]
    return df

def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding="latin1")
    df = clean_columns(df)

    needed = ["Test Time (s)", "Current (A)", "Voltage (V)"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=needed).reset_index(drop=True)
    df["Test Time (s)"] = df["Test Time (s)"].astype(float)
    df["Current (A)"] = df["Current (A)"].astype(float)
    df["Voltage (V)"] = df["Voltage (V)"].astype(float)
    return df

def contiguous_segments(mask):
    segments = []
    in_seg = False
    start = None
    for k, val in enumerate(mask):
        if val and not in_seg:
            in_seg = True
            start = k
        elif not val and in_seg:
            in_seg = False
            segments.append((start, k - 1))
    if in_seg:
        segments.append((start, len(mask) - 1))
    return segments

def find_full_charge_rest_segments(t, i, v):
    rest_mask = np.abs(i) <= REST_CURRENT_THRESHOLD
    rest_segments = contiguous_segments(rest_mask)

    long_rests = []
    for s, e in rest_segments:
        duration = t[e] - t[s]
        if duration >= MIN_FULL_REST_DURATION_S:
            mean_v = float(np.mean(v[s:e + 1]))
            long_rests.append((s, e, duration, mean_v))

    if not long_rests:
        return []

    max_rest_v = max(seg[3] for seg in long_rests)
    full_rests = [seg for seg in long_rests if seg[3] >= max_rest_v - FULL_VOLTAGE_WINDOW_V]
    return full_rests

def choose_between_full_charges_window(t, i, v):
    full_rests = find_full_charge_rest_segments(t, i, v)

    if len(full_rests) >= 2:
        first = full_rests[0]
        last = full_rests[-1]
        start_time = max(t[first[0]] - EDGE_BUFFER_S, t[0])
        end_time = min(t[last[1]] + EDGE_BUFFER_S, t[-1])
        start_idx = int(np.searchsorted(t, start_time, side="left"))
        end_idx = int(np.searchsorted(t, end_time, side="right") - 1)
        method = "full-charge rests"
        return start_idx, end_idx, full_rests, method

    active = np.where(np.abs(i) >= ACTIVE_CURRENT_THRESHOLD)[0]
    if len(active) == 0:
        raise ValueError("Could not find any active UDDS current region.")

    start_time = max(t[int(active[0])] - EDGE_BUFFER_S, t[0])
    end_time = min(t[int(active[-1])] + EDGE_BUFFER_S, t[-1])
    start_idx = int(np.searchsorted(t, start_time, side="left"))
    end_idx = int(np.searchsorted(t, end_time, side="right") - 1)
    method = "active-current fallback"
    return start_idx, end_idx, full_rests, method

def compute_eta_and_qd(t, i):
    q_dis_as = 0.0
    q_chg_as = 0.0
    for k in range(len(t) - 1):
        dt = max(t[k + 1] - t[k], 0.0)
        ik = i[k]
        if ik < 0:
            q_dis_as += (-ik) * dt
        elif ik > 0:
            q_chg_as += ik * dt

    if q_dis_as <= 0:
        raise ValueError("Computed discharge throughput is zero.")
    if q_chg_as <= 0:
        raise ValueError("Computed charge throughput is zero.")

    eta = q_dis_as / q_chg_as
    return q_dis_as, q_chg_as, eta

def build_soc_label(t, i, eta, qd_as):
    soc = np.zeros(len(t), dtype=float)
    soc[0] = 1.0

    for k in range(len(t) - 1):
        dt = max(t[k + 1] - t[k], 0.0)
        ik = i[k]
        if ik > 0:
            soc[k + 1] = soc[k] + eta * ik * dt / qd_as
        else:
            soc[k + 1] = soc[k] + ik * dt / qd_as
        soc[k + 1] = np.clip(soc[k + 1], 0.0, 1.0)

    return soc

def main():
    csv_path = find_existing_file(CSV_CANDIDATES)
    df = load_data(csv_path)

    t_all = df["Test Time (s)"].to_numpy(dtype=float)
    i_all = df["Current (A)"].to_numpy(dtype=float)
    v_all = df["Voltage (V)"].to_numpy(dtype=float)

    start_idx, end_idx, full_rests, method = choose_between_full_charges_window(t_all, i_all, v_all)
    seg = df.iloc[start_idx:end_idx + 1].copy().reset_index(drop=True)
    seg["Test Time (s)"] = seg["Test Time (s)"].astype(float) - float(seg["Test Time (s)"].iloc[0])

    t = seg["Test Time (s)"].to_numpy(dtype=float)
    i = seg["Current (A)"].to_numpy(dtype=float)
    v = seg["Voltage (V)"].to_numpy(dtype=float)

    qd_as, qchg_as, eta = compute_eta_and_qd(t, i)
    soc_label = build_soc_label(t, i, eta, qd_as)

    seg["SOC_label"] = soc_label
    seg["Current_abs_A"] = np.abs(i)

    out_csv = OUTPUT_DIR / "udds_between_full_charges_with_soc.csv"
    seg.to_csv(out_csv, index=False)

    summary_lines = [
        f"source_csv: {csv_path.name}",
        f"selection_method: {method}",
        f"start_idx_in_source: {start_idx}",
        f"end_idx_in_source: {end_idx}",
        f"duration_s: {t[-1] if len(t) else 0.0:.6f}",
        f"Qd (A*s): {qd_as:.10f}",
        f"Qd (Ah): {qd_as / 3600.0:.10f}",
        f"Qcharge (A*s): {qchg_as:.10f}",
        f"Qcharge (Ah): {qchg_as / 3600.0:.10f}",
        f"eta: {eta:.10f}",
        f"initial_SOC_label: {soc_label[0]:.10f}",
        f"final_SOC_label: {soc_label[-1]:.10f}",
        f"min_SOC_label: {np.min(soc_label):.10f}",
        f"max_SOC_label: {np.max(soc_label):.10f}",
        f"num_rows: {len(seg)}",
        f"num_full_charge_rest_candidates: {len(full_rests)}",
    ]

    out_txt = OUTPUT_DIR / "udds_soc_summary.txt"
    out_txt.write_text("\n".join(summary_lines))

    print(f"Using raw UDDS CSV: {csv_path.name}")
    print(f"Selection method: {method}")
    print(f"Segment rows: {len(seg)}")
    print(f"Segment duration: {t[-1]:.2f} s")
    print(f"Qd = {qd_as:.6f} A*s = {qd_as / 3600.0:.6f} Ah")
    print(f"Qcharge = {qchg_as:.6f} A*s = {qchg_as / 3600.0:.6f} Ah")
    print(f"eta = {eta:.8f}")
    print(f"SOC range: min={np.min(soc_label):.6f}, max={np.max(soc_label):.6f}, final={soc_label[-1]:.6f}")
    print(f"Saved: {out_csv.name}")
    print(f"Saved: {out_txt.name}")

if __name__ == "__main__":
    main()

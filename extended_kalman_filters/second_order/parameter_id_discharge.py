import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "second_order"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "second_order"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_CANDIDATES = [
    "SAMSUNG_Cell_5_incremental_01C_Channel_6_Wb_1 (1).CSV",
    "SAMSUNG_Cell_5_incremental_01C_Channel_6_Wb_1 (1) copy.CSV",
    "SAMSUNG_Cell_5_incremental_01C_Channel_6_Wb_1.CSV",
]

REST_CURRENT_THRESHOLD = 0.005
ACTIVE_CURRENT_THRESHOLD = 0.05
MIN_REST_DURATION = 120.0
FIT_WINDOW = 300.0

MAX_OCV_POLY_ORDER = 5
MAX_PARAM_POLY_ORDER = 4

def find_existing_file(candidates):
    for name in candidates:
        for base in (DATA_DIR, MODULE_DIR):
            path = base / name
            if path.exists():
                return path
    matches = sorted(DATA_DIR.glob("SAMSUNG_Cell_5_incremental_01C_Channel_6_Wb_1*.CSV")) or sorted(MODULE_DIR.glob("SAMSUNG_Cell_5_incremental_01C_Channel_6_Wb_1*.CSV"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Could not find the incremental CSV. Expected one of: " + ", ".join(candidates)
    )

def clean_columns(df):
    df = df.copy()
    df.columns = [c.replace("ÿ", "").replace("ï»¿", "").strip() for c in df.columns]
    return df

def exp_relax_discharge_2rc(t, v_inf, A1, tau1, A2, tau2):
    return v_inf - A1 * np.exp(-t / tau1) - A2 * np.exp(-t / tau2)

def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding="latin1")
    df = clean_columns(df)

    needed = ["Test Time (s)", "Current (A)", "Voltage (V)", "Discharge Capacity (Ah)"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=["Test Time (s)", "Current (A)", "Voltage (V)"]).reset_index(drop=True)

    mid = len(df) // 2
    df = df.iloc[:mid].reset_index(drop=True)

    t0 = float(df["Test Time (s)"].iloc[0])
    df["Test Time (s)"] = df["Test Time (s)"].astype(float) - t0

    t = df["Test Time (s)"].to_numpy(dtype=float)
    i = df["Current (A)"].to_numpy(dtype=float)
    v = df["Voltage (V)"].to_numpy(dtype=float)
    dcap = np.nan_to_num(df["Discharge Capacity (Ah)"].to_numpy(dtype=float), nan=0.0)
    return df, t, i, v, dcap

def find_main_discharge_window(i):
    active = np.where(i < -ACTIVE_CURRENT_THRESHOLD)[0]
    if len(active) == 0:
        raise ValueError("Could not find discharge period from current.")
    return int(active[0]), int(active[-1])

def coulomb_count_soc_discharge(t, i, start_idx, end_idx):
    soc = np.full_like(t, np.nan, dtype=float)
    soc[start_idx] = 1.0

    discharged_ah = 0.0
    discharged_history = np.zeros_like(t, dtype=float)
    for k in range(start_idx, end_idx):
        dt = max(t[k + 1] - t[k], 0.0)
        ik = max(-i[k], 0.0)
        discharged_ah += ik * dt / 3600.0
        discharged_history[k + 1] = discharged_ah

    total_ah = discharged_history[end_idx]
    if total_ah <= 0:
        raise ValueError("Total discharged Ah computed as zero.")

    for k in range(start_idx, end_idx + 1):
        soc[k] = 1.0 - discharged_history[k] / total_ah

    return soc, total_ah

def contiguous_segments(mask):
    segments = []
    in_seg = False
    seg_start = None
    for k, val in enumerate(mask):
        if val and not in_seg:
            in_seg = True
            seg_start = k
        elif not val and in_seg:
            in_seg = False
            segments.append((seg_start, k - 1))
    if in_seg:
        segments.append((seg_start, len(mask) - 1))
    return segments

def find_rest_segments(t, i, start_idx, end_idx):
    mask = np.zeros_like(i, dtype=bool)
    mask[start_idx:end_idx + 1] = np.abs(i[start_idx:end_idx + 1]) <= REST_CURRENT_THRESHOLD
    segments = contiguous_segments(mask)
    return [(s, e) for s, e in segments if (t[e] - t[s]) >= MIN_REST_DURATION]

def estimate_r0(t, i, v, rest_start_idx, pre_pts=5, post_pts=5):
    s = rest_start_idx
    if s < pre_pts + 1 or s + post_pts >= len(t):
        return np.nan, np.nan, np.nan, np.nan

    i_before = np.mean(i[s - pre_pts:s])
    i_after = np.mean(i[s:s + post_pts])
    v_before = np.mean(v[s - pre_pts:s])
    v_after = np.mean(v[s:s + post_pts])

    delta_i = i_after - i_before
    delta_v = v_after - v_before
    if abs(delta_i) < 1e-12:
        return np.nan, i_before, v_before, v_after

    return abs(delta_v / delta_i), i_before, v_before, v_after

def sort_branches(A1, tau1, A2, tau2):
    if tau1 <= tau2:
        return A1, tau1, A2, tau2
    return A2, tau2, A1, tau1

def fit_rest_segment_discharge_2rc(t, i, v, seg_start, seg_end):
    t0 = t[seg_start]
    idx = np.where(
        (t >= t0)
        & (t <= t0 + FIT_WINDOW)
        & (np.arange(len(t)) >= seg_start)
        & (np.arange(len(t)) <= seg_end)
    )[0]
    if len(idx) < 15:
        return None

    tt = t[idx] - t0
    vv = v[idx]

    r0, i_before, v_before, v_after = estimate_r0(t, i, v, seg_start)
    if np.isnan(r0):
        return None

    delta_i = abs(i_before)
    if delta_i < ACTIVE_CURRENT_THRESHOLD:
        return None

    v_inf_guess = np.mean(vv[-min(10, len(vv)):])
    total_amp = max(v_inf_guess - vv[0], 1e-5)
    A1_guess = 0.65 * total_amp
    A2_guess = 0.35 * total_amp
    tau1_guess = 10.0
    tau2_guess = 120.0
    amp_upper = max(1.0, 5.0 * total_amp)

    try:
        popt, _ = curve_fit(
            exp_relax_discharge_2rc,
            tt,
            vv,
            p0=[v_inf_guess, A1_guess, tau1_guess, A2_guess, tau2_guess],
            bounds=(
                [min(vv) - 0.2, 0.0, 0.5, 0.0, 1.0],
                [max(vv) + 0.2, amp_upper, 5000.0, amp_upper, 20000.0],
            ),
            maxfev=50000,
        )
    except Exception:
        return None

    v_inf, A1, tau1, A2, tau2 = popt
    A1, tau1, A2, tau2 = sort_branches(A1, tau1, A2, tau2)

    r1 = A1 / delta_i if delta_i > 0 else np.nan
    r2 = A2 / delta_i if delta_i > 0 else np.nan
    c1 = tau1 / r1 if r1 > 0 else np.nan
    c2 = tau2 / r2 if r2 > 0 else np.nan

    vv_fit = exp_relax_discharge_2rc(tt, v_inf, A1, tau1, A2, tau2)
    rmse = np.sqrt(np.mean((vv - vv_fit) ** 2))

    return {
        "rest_start_idx": seg_start,
        "rest_end_idx": seg_end,
        "rest_start_time_s": t[seg_start],
        "rest_end_time_s": t[seg_end],
        "I_before_A": i_before,
        "R0_ohm": r0,
        "R1_ohm": r1,
        "C1_F": c1,
        "tau1_s": tau1,
        "R2_ohm": r2,
        "C2_F": c2,
        "tau2_s": tau2,
        "OCV_V": v_inf,
        "A1_V": A1,
        "A2_V": A2,
        "V_before_V": v_before,
        "V_after_V": v_after,
        "fit_rmse_V": rmse,
    }

def fit_poly_adaptive(x, y, max_order):
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    if len(x) < 2:
        return None, None
    order = min(max_order, len(x) - 1)
    coeffs = np.polyfit(x, y, order)
    return coeffs, order

def main():
    csv_path = find_existing_file(CSV_CANDIDATES)
    _, t, i, v, _ = load_data(csv_path)

    discharge_start, discharge_end = find_main_discharge_window(i)
    soc, total_ah = coulomb_count_soc_discharge(t, i, discharge_start, discharge_end)
    rests = find_rest_segments(t, i, discharge_start, discharge_end)

    rows = []
    for seg_num, (s, e) in enumerate(rests, start=1):
        result = fit_rest_segment_discharge_2rc(t, i, v, s, e)
        if result is None:
            continue
        result["SOC"] = soc[s]
        result["segment_number"] = seg_num
        rows.append(result)

    params_df = pd.DataFrame(rows)
    if len(params_df) == 0:
        raise ValueError("No valid discharge relaxation segments were fitted.")

    params_df = params_df.replace([np.inf, -np.inf], np.nan)
    params_df = params_df.dropna(
        subset=["SOC", "OCV_V", "R0_ohm", "R1_ohm", "C1_F", "R2_ohm", "C2_F"]
    ).reset_index(drop=True)
    params_df = params_df[(params_df["SOC"] >= 0.0) & (params_df["SOC"] <= 1.0)].reset_index(drop=True)
    params_df = params_df[
        (params_df["R0_ohm"] > 0) & (params_df["R0_ohm"] < 1.0) &
        (params_df["R1_ohm"] > 1e-6) & (params_df["R1_ohm"] < 1.0) &
        (params_df["R2_ohm"] > 1e-6) & (params_df["R2_ohm"] < 1.0) &
        (params_df["C1_F"] > 1.0) & (params_df["C1_F"] < 1e9) &
        (params_df["C2_F"] > 1.0) & (params_df["C2_F"] < 1e10) &
        (params_df["tau1_s"] > 0.1) & (params_df["tau1_s"] < 10000.0) &
        (params_df["tau2_s"] > 0.1) & (params_df["tau2_s"] < 50000.0) &
        (params_df["tau2_s"] >= params_df["tau1_s"])
    ].sort_values("SOC").reset_index(drop=True)

    if len(params_df) == 0:
        raise ValueError("All fitted discharge rows were filtered out as nonphysical.")

    params_df.to_csv(OUTPUT_DIR / "identified_discharge_params_table_2rc.csv", index=False)

    z = params_df["SOC"].to_numpy()
    ocv_vals = params_df["OCV_V"].to_numpy()
    r0_vals = params_df["R0_ohm"].to_numpy()
    r1_vals = params_df["R1_ohm"].to_numpy()
    c1_vals = params_df["C1_F"].to_numpy()
    r2_vals = params_df["R2_ohm"].to_numpy()
    c2_vals = params_df["C2_F"].to_numpy()

    ocv_coeffs, ocv_order = fit_poly_adaptive(z, ocv_vals, MAX_OCV_POLY_ORDER)
    r0_coeffs, r0_order = fit_poly_adaptive(z, r0_vals, MAX_PARAM_POLY_ORDER)
    r1_coeffs, r1_order = fit_poly_adaptive(z, r1_vals, MAX_PARAM_POLY_ORDER)
    c1_coeffs, c1_order = fit_poly_adaptive(z, c1_vals, MAX_PARAM_POLY_ORDER)
    r2_coeffs, r2_order = fit_poly_adaptive(z, r2_vals, MAX_PARAM_POLY_ORDER)
    c2_coeffs, c2_order = fit_poly_adaptive(z, c2_vals, MAX_PARAM_POLY_ORDER)

    np.savez(
        OUTPUT_DIR / "identified_discharge_param_fits_2rc.npz",
        ocv_coeffs=np.array([]) if ocv_coeffs is None else ocv_coeffs,
        r0_coeffs=np.array([]) if r0_coeffs is None else r0_coeffs,
        r1_coeffs=np.array([]) if r1_coeffs is None else r1_coeffs,
        c1_coeffs=np.array([]) if c1_coeffs is None else c1_coeffs,
        r2_coeffs=np.array([]) if r2_coeffs is None else r2_coeffs,
        c2_coeffs=np.array([]) if c2_coeffs is None else c2_coeffs,
        ocv_order=-1 if ocv_order is None else ocv_order,
        r0_order=-1 if r0_order is None else r0_order,
        r1_order=-1 if r1_order is None else r1_order,
        c1_order=-1 if c1_order is None else c1_order,
        r2_order=-1 if r2_order is None else r2_order,
        c2_order=-1 if c2_order is None else c2_order,
        total_discharged_ah=total_ah,
        source_csv=str(csv_path.name),
    )

    print(f"Using CSV: {csv_path.name}")
    print(f"Discharge window: start={discharge_start}, end={discharge_end}")
    print(f"Total discharged capacity: {total_ah:.6f} Ah")
    print(f"Valid discharge 2RC segments kept: {len(params_df)}")
    print("Saved identified_discharge_params_table_2rc.csv")
    print("Saved identified_discharge_param_fits_2rc.npz")

if __name__ == "__main__":
    main()

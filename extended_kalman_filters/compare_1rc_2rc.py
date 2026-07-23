from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
FIRST_ORDER_DIR = BASE_DIR / "outputs" / "first_order"
SECOND_ORDER_DIR = BASE_DIR / "outputs" / "second_order"
OUTPUT_DIR = BASE_DIR / "outputs" / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def find_first_existing(candidates, label):
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {label}.")

def pick_column(df, candidates, label, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find a column for {label}. Tried: {candidates}")
    return None

def load_results():
    first_order_csv = find_first_existing([
        FIRST_ORDER_DIR / "udds_1rc_ekf_results.csv",
        FIRST_ORDER_DIR / "udds_ekf_first_order_avg_results.csv",
        FIRST_ORDER_DIR / "udds_1rc_ekf_results.csv",
        FIRST_ORDER_DIR / "udds_ekf_results.csv",
    ], "a first-order EKF results CSV in outputs/first_order/")
    second_order_csv = find_first_existing([
        SECOND_ORDER_DIR / "udds_2rc_ekf_results.csv",
        SECOND_ORDER_DIR / "udds_ekf_rc2_results.csv",
        SECOND_ORDER_DIR / "udds_ekf_results.csv",
    ], "a second-order EKF results CSV in outputs/second_order/")
    df1 = pd.read_csv(first_order_csv)
    df2 = pd.read_csv(second_order_csv)
    print(f"Loaded 1RC results: {first_order_csv}")
    print(f"Loaded 2RC results: {second_order_csv}")
    return df1, df2

def build_standardized_df(df, model_name):
    time_col = pick_column(df, ["Time_s", "Test Time (s)", "time_s", "time"], f"time for {model_name}")
    current_col = pick_column(df, ["Current_A", "Current (A)", "current_a", "current"], f"current for {model_name}")
    voltage_meas_col = pick_column(df, ["Voltage_V", "Voltage (V)", "voltage_v", "voltage"], f"measured voltage for {model_name}")
    soc_est_col = pick_column(df, ["SOC_EKF", "soc_est", "SOC_est", "z_est", "soc_ekf"], f"soc estimate for {model_name}")
    soc_label_col = pick_column(df, ["SOC_label", "soc_label", "SOC_true", "soc_true", "SoC"], f"soc label for {model_name}")
    voltage_pred_col = pick_column(df, ["Voltage_pred_EKF", "v_pred", "Voltage_pred", "voltage_pred", "Voltage_pred_EKF_2RC"], f"predicted voltage for {model_name}")
    residual_col = pick_column(df, ["Voltage_residual_EKF", "residual", "Residual_V", "voltage_residual", "Voltage_residual_EKF_2RC"], f"residual for {model_name}")
    sigma_col = pick_column(df, ["SOC_sigma", "soc_sigma"], f"sigma for {model_name}", required=False)
    ci_low_col = pick_column(df, ["SOC_CI_low", "soc_ci_low"], f"CI low for {model_name}", required=False)
    ci_high_col = pick_column(df, ["SOC_CI_high", "soc_ci_high"], f"CI high for {model_name}", required=False)

    out = pd.DataFrame({
        "time_s": pd.to_numeric(df[time_col], errors="coerce"),
        "current_a": pd.to_numeric(df[current_col], errors="coerce"),
        "voltage_meas_v": pd.to_numeric(df[voltage_meas_col], errors="coerce"),
        "soc_est": pd.to_numeric(df[soc_est_col], errors="coerce"),
        "soc_label": pd.to_numeric(df[soc_label_col], errors="coerce"),
        "voltage_pred_v": pd.to_numeric(df[voltage_pred_col], errors="coerce"),
        "residual_v": pd.to_numeric(df[residual_col], errors="coerce"),
        "soc_sigma": np.nan if sigma_col is None else pd.to_numeric(df[sigma_col], errors="coerce"),
        "soc_ci_low": np.nan if ci_low_col is None else pd.to_numeric(df[ci_low_col], errors="coerce"),
        "soc_ci_high": np.nan if ci_high_col is None else pd.to_numeric(df[ci_high_col], errors="coerce"),
    }).dropna(subset=["time_s","current_a","voltage_meas_v","soc_est","soc_label","voltage_pred_v","residual_v"]).reset_index(drop=True)

    if out["soc_ci_low"].isna().all() or out["soc_ci_high"].isna().all():
        sigma = out["soc_sigma"].to_numpy(dtype=float)
        if np.isfinite(sigma).any():
            out["soc_ci_low"] = np.clip(out["soc_est"] - 1.96 * sigma, 0.0, 1.0)
            out["soc_ci_high"] = np.clip(out["soc_est"] + 1.96 * sigma, 0.0, 1.0)
    return out

def summarize(df_std, model_name):
    return {
        "model": model_name,
        "n_samples": int(len(df_std)),
        "soc_rmse": float(np.sqrt(np.mean((df_std["soc_est"] - df_std["soc_label"]) ** 2))),
        "soc_mae": float(np.mean(np.abs(df_std["soc_est"] - df_std["soc_label"]))),
        "voltage_rmse_v": float(np.sqrt(np.mean((df_std["voltage_pred_v"] - df_std["voltage_meas_v"]) ** 2))),
        "voltage_mae_v": float(np.mean(np.abs(df_std["voltage_pred_v"] - df_std["voltage_meas_v"]))),
        "residual_rmse_v": float(np.sqrt(np.mean(df_std["residual_v"] ** 2))),
        "residual_mae_v": float(np.mean(np.abs(df_std["residual_v"]))),
    }

def save_summary(rows):
    out_path = OUTPUT_DIR / "ekf_comparison_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

def plot_soc(df1, df2):
    plt.figure(figsize=(12, 6))
    plt.plot(df1["time_s"], df1["soc_label"], label="SOC label", linewidth=2)
    plt.plot(df1["time_s"], df1["soc_est"], label="1RC EKF SOC", linewidth=2)
    if np.isfinite(df1["soc_ci_low"]).any():
        plt.fill_between(df1["time_s"], df1["soc_ci_low"], df1["soc_ci_high"], alpha=0.15, label="1RC 95% CI")
    plt.plot(df2["time_s"], df2["soc_est"], label="2RC EKF SOC", linewidth=2)
    if np.isfinite(df2["soc_ci_low"]).any():
        plt.fill_between(df2["time_s"], df2["soc_ci_low"], df2["soc_ci_high"], alpha=0.12, label="2RC 95% CI")
    plt.xlabel("Time (s)")
    plt.ylabel("SOC")
    plt.title("EKF SOC Comparison")
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = OUTPUT_DIR / "ekf_soc_comparison.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved: {out_path}")

def plot_voltage(df1, df2):
    plt.figure(figsize=(12, 6))
    plt.plot(df1["time_s"], df1["voltage_meas_v"], label="Measured voltage")
    plt.plot(df1["time_s"], df1["voltage_pred_v"], label="1RC predicted voltage")
    plt.plot(df2["time_s"], df2["voltage_pred_v"], label="2RC predicted voltage")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("EKF Voltage Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = OUTPUT_DIR / "ekf_voltage_comparison.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved: {out_path}")

def plot_residual(df1, df2):
    plt.figure(figsize=(12, 6))
    plt.plot(df1["time_s"], df1["residual_v"], label="1RC residual")
    plt.plot(df2["time_s"], df2["residual_v"], label="2RC residual")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage residual (V)")
    plt.title("EKF Residual Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = OUTPUT_DIR / "ekf_residual_comparison.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    df1_raw, df2_raw = load_results()
    df1 = build_standardized_df(df1_raw, "1RC")
    df2 = build_standardized_df(df2_raw, "2RC")
    n = min(len(df1), len(df2))
    if n == 0:
        raise ValueError("One of the EKF result files has no usable rows.")
    if len(df1) != len(df2):
        print(f"Row count mismatch detected. Truncating to n={n}")
    df1 = df1.iloc[:n].reset_index(drop=True)
    df2 = df2.iloc[:n].reset_index(drop=True)
    save_summary([summarize(df1, "1RC"), summarize(df2, "2RC")])
    plot_soc(df1, df2)
    plot_voltage(df1, df2)
    plot_residual(df1, df2)
    print("Done.")

if __name__ == "__main__":
    main()

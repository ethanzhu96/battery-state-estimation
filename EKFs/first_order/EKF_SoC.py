import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "first_order"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "first_order"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UDDS_CSV_CANDIDATES = [
    OUTPUT_DIR / "udds_between_full_charges_with_soc.csv",
    DATA_DIR / "udds_between_full_charges_with_soc.csv",
    MODULE_DIR / "udds_between_full_charges_with_soc.csv",
]
UDDS_SUMMARY_CANDIDATES = [
    OUTPUT_DIR / "udds_soc_summary.txt",
    DATA_DIR / "udds_soc_summary.txt",
    MODULE_DIR / "udds_soc_summary.txt",
]
DISCHARGE_FIT_CANDIDATES = [
    OUTPUT_DIR / "identified_discharge_param_fits.npz",
    DATA_DIR / "identified_discharge_param_fits.npz",
    MODULE_DIR / "identified_discharge_param_fits.npz",
]
CHARGE_FIT_CANDIDATES = [
    OUTPUT_DIR / "identified_charge_param_fits.npz",
    DATA_DIR / "identified_charge_param_fits.npz",
    MODULE_DIR / "identified_charge_param_fits.npz",
]

def find_first_existing(candidates, label):
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing {label}. Tried:\n{tried}")

def clip_soc(z):
    return float(np.clip(z, 0.0, 1.0))

def polyval(coeffs, x):
    return float(np.polyval(coeffs, x))

def polyderval(coeffs, x):
    return float(np.polyval(np.polyder(coeffs), x))

def read_summary_value(summary_text, key):
    for line in summary_text.splitlines():
        if line.strip().startswith(key):
            parts = line.split(":")
            if len(parts) >= 2:
                return float(parts[1].strip())
    raise ValueError(f"Could not find '{key}' in udds_soc_summary.txt")

def load_udds_data():
    udds_csv = find_first_existing(UDDS_CSV_CANDIDATES, "UDDS SOC-labeled CSV")
    df = pd.read_csv(udds_csv)
    soc_col = "SOC_label" if "SOC_label" in df.columns else "SoC"
    needed = ["Test Time (s)", "Current (A)", "Voltage (V)", soc_col]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column in UDDS CSV: {col}")
    t = df["Test Time (s)"].to_numpy(dtype=float)
    i = df["Current (A)"].to_numpy(dtype=float)
    v = df["Voltage (V)"].to_numpy(dtype=float)
    soc_label = df[soc_col].to_numpy(dtype=float)
    summary_path = find_first_existing(UDDS_SUMMARY_CANDIDATES, "UDDS SOC summary")
    summary_text = summary_path.read_text()
    eta = read_summary_value(summary_text, "eta")
    qd_as = read_summary_value(summary_text, "Qd (A*s)")
    return df, t, i, v, soc_label, eta, qd_as

def load_model():
    discharge_fits = find_first_existing(DISCHARGE_FIT_CANDIDATES, "1RC discharge parameter fit NPZ")
    charge_fits = find_first_existing(CHARGE_FIT_CANDIDATES, "1RC charge parameter fit NPZ")
    dis_fit = np.load(discharge_fits)
    chg_fit = np.load(charge_fits)
    required = ["ocv_coeffs", "r0_coeffs", "r1_coeffs", "c1_coeffs"]
    for key in required:
        if key not in dis_fit or key not in chg_fit:
            raise ValueError(f"Missing '{key}' in one of the 1RC fit files.")
        if len(dis_fit[key]) == 0 or len(chg_fit[key]) == 0:
            raise ValueError(f"Empty '{key}' in one of the 1RC fit files.")
    return {
        "ocv_dis": np.array(dis_fit["ocv_coeffs"], dtype=float),
        "ocv_chg": np.array(chg_fit["ocv_coeffs"], dtype=float),
        "r0_dis": np.array(dis_fit["r0_coeffs"], dtype=float),
        "r0_chg": np.array(chg_fit["r0_coeffs"], dtype=float),
        "r1_dis": np.array(dis_fit["r1_coeffs"], dtype=float),
        "r1_chg": np.array(chg_fit["r1_coeffs"], dtype=float),
        "c1_dis": np.array(dis_fit["c1_coeffs"], dtype=float),
        "c1_chg": np.array(chg_fit["c1_coeffs"], dtype=float),
    }

def avg_eval(z, dis_coeffs, chg_coeffs):
    z = clip_soc(z)
    return 0.5 * (polyval(dis_coeffs, z) + polyval(chg_coeffs, z))

def avg_param(z, dis_coeffs, chg_coeffs, floor):
    return max(avg_eval(z, dis_coeffs, chg_coeffs), floor)

def ocv_avg(z, model):
    return avg_eval(z, model["ocv_dis"], model["ocv_chg"])

def docv_avg(z, model):
    z = clip_soc(z)
    d = 0.5 * (polyderval(model["ocv_dis"], z) + polyderval(model["ocv_chg"], z))
    return float(np.clip(d, 1e-4, 5.0))

def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def validate_ekf_inputs(t, i, v_meas, soc_label, qd_as):
    lengths = {len(t), len(i), len(v_meas), len(soc_label)}
    if len(lengths) != 1:
        raise ValueError("EKF inputs must have matching lengths.")
    if len(t) < 2:
        raise ValueError("EKF needs at least two samples.")
    if not np.isfinite(qd_as) or qd_as <= 0:
        raise ValueError("Qd_as must be finite and positive.")
    for name, arr in [("t", t), ("i", i), ("v_meas", v_meas), ("soc_label", soc_label)]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"EKF input '{name}' contains non-finite values.")

def run_ekf(t, i, v_meas, soc_label, eta, qd_as, model,
            meas_var=1e-2, process_vars=(1e-12, 1e-5), residual_gate=0.02,
            initial_cov=(1e-8, 1e-3)):
    validate_ekf_inputs(t, i, v_meas, soc_label, qd_as)
    n = len(t)
    xhat = np.zeros((n, 2), dtype=float)
    phat = np.zeros((n, 2, 2), dtype=float)
    v_pred_hist = np.zeros(n, dtype=float)
    resid_hist = np.zeros(n, dtype=float)
    k_soc_hist = np.zeros(n, dtype=float)

    # Trust coulomb-counted SOC strongly; let the RC state absorb voltage transients.
    xhat[0] = np.array([clip_soc(soc_label[0]), 0.0], dtype=float)
    phat[0] = np.diag(initial_cov)
    q_soc, q_u1 = process_vars
    Qk = np.diag([q_soc, q_u1])

    for k in range(n - 1):
        dt = max(float(t[k + 1] - t[k]), 0.0)
        ik = float(i[k])
        i_next = float(i[k + 1])
        v_next = float(v_meas[k + 1])
        soc_k, u1_k = xhat[k]

        if ik > 0:
            soc_pred = soc_k + eta * ik * dt / qd_as
        else:
            soc_pred = soc_k + ik * dt / qd_as
        soc_pred = clip_soc(soc_pred)

        r0 = avg_param(soc_pred, model["r0_dis"], model["r0_chg"], 1e-5)
        r1 = avg_param(soc_pred, model["r1_dis"], model["r1_chg"], 1e-5)
        c1 = avg_param(soc_pred, model["c1_dis"], model["c1_chg"], 1.0)

        tau1 = max(r1 * c1, 1e-4)
        a1 = np.exp(-dt / tau1)
        b1 = r1 * (1.0 - a1)
        u1_pred = a1 * u1_k + b1 * ik

        x_pred = np.array([soc_pred, u1_pred], dtype=float)
        F = np.array([[1.0, 0.0], [0.0, a1]], dtype=float)
        P_pred = F @ phat[k] @ F.T + Qk

        v_pred = ocv_avg(x_pred[0], model) + x_pred[1] + i_next * r0
        y = v_next - v_pred

        adaptive_R = meas_var + 1e-4 * min(abs(i_next), 5.0)
        Rk = np.array([[adaptive_R]], dtype=float)

        if abs(y) > residual_gate:
            x_upd = x_pred.copy()
            P_upd = P_pred.copy()
            K = np.zeros((2, 1), dtype=float)
        else:
            d_ocv = docv_avg(x_pred[0], model)
            H = np.array([[d_ocv, 1.0]], dtype=float)
            S = H @ P_pred @ H.T + Rk
            K = (P_pred @ H.T) / float(S[0, 0])
            x_upd = x_pred + K.flatten() * y
            x_upd[0] = clip_soc(x_upd[0])
            I2 = np.eye(2)
            P_upd = (I2 - K @ H) @ P_pred @ (I2 - K @ H).T + K @ Rk @ K.T

        xhat[k + 1] = x_upd
        phat[k + 1] = P_upd
        v_pred_hist[k + 1] = v_pred
        resid_hist[k + 1] = y
        k_soc_hist[k + 1] = float(K[0, 0])

    r0_0 = avg_param(xhat[0, 0], model["r0_dis"], model["r0_chg"], 1e-5)
    v_pred_hist[0] = ocv_avg(xhat[0, 0], model) + xhat[0, 1] + i[0] * r0_0
    resid_hist[0] = v_meas[0] - v_pred_hist[0]
    return xhat, phat, v_pred_hist, resid_hist, k_soc_hist

def plot_soc_results(t_rel, soc_label, soc_est, P_hat):
    sigma = np.sqrt(np.maximum(P_hat[:, 0, 0], 0.0))
    lo = np.clip(soc_est - 1.96 * sigma, 0.0, 1.0)
    hi = np.clip(soc_est + 1.96 * sigma, 0.0, 1.0)
    plt.figure(figsize=(11, 5))
    plt.plot(t_rel, soc_label, label="SOC label", linewidth=2)
    plt.plot(t_rel, soc_est, label="1RC EKF SOC", linewidth=2)
    plt.fill_between(t_rel, lo, hi, alpha=0.18, label="95% CI")
    plt.xlabel("Time since first full charge (s)")
    plt.ylabel("SOC")
    plt.title("1RC EKF: SOC label vs EKF")
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_voltage_results(t_rel, v_meas, v_pred):
    plt.figure(figsize=(11, 5))
    plt.plot(t_rel, v_meas, label="Measured voltage")
    plt.plot(t_rel, v_pred, label="1RC predicted voltage")
    plt.xlabel("Time since first full charge (s)")
    plt.ylabel("Voltage (V)")
    plt.title("1RC EKF: Measured vs predicted voltage")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_residual(t_rel, resid):
    plt.figure(figsize=(11, 4))
    plt.plot(t_rel, resid)
    plt.xlabel("Time since first full charge (s)")
    plt.ylabel("Voltage residual (V)")
    plt.title("1RC EKF: Voltage residual")
    plt.grid(True)
    plt.tight_layout()

def main():
    df, t, i, v_meas, soc_label, eta, qd_as = load_udds_data()
    model = load_model()
    xhat, phat, v_pred_hist, resid_hist, k_soc_hist = run_ekf(
        t=t, i=i, v_meas=v_meas, soc_label=soc_label, eta=eta, qd_as=qd_as, model=model
    )
    soc_est = xhat[:, 0]
    soc_sigma = np.sqrt(np.maximum(phat[:, 0, 0], 0.0))
    t_rel = t - t[0]

    print("====================================")
    print("1RC EKF RESULTS")
    print("====================================")
    print(f"SOC RMSE     = {rmse(soc_est, soc_label):.8f}")
    print(f"Voltage RMSE = {rmse(v_pred_hist, v_meas):.8f} V")
    print(f"Initial SOC label = {soc_label[0]:.6f}")
    print(f"Initial SOC EKF   = {soc_est[0]:.6f}")
    print(f"Final SOC label   = {soc_label[-1]:.6f}")
    print(f"Final SOC EKF     = {soc_est[-1]:.6f}")
    print("====================================")

    out_df = df.copy()
    out_df["SOC_EKF"] = soc_est
    out_df["SOC_sigma"] = soc_sigma
    out_df["SOC_CI_low"] = np.clip(soc_est - 1.96 * soc_sigma, 0.0, 1.0)
    out_df["SOC_CI_high"] = np.clip(soc_est + 1.96 * soc_sigma, 0.0, 1.0)
    out_df["Voltage_pred_EKF"] = v_pred_hist
    out_df["Voltage_residual_EKF"] = resid_hist
    out_df["K_soc_EKF"] = k_soc_hist

    out_csv = OUTPUT_DIR / "udds_1rc_ekf_results.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv.name}")

    plot_soc_results(t_rel, soc_label, soc_est, phat)
    plt.savefig(OUTPUT_DIR / "ekf_1rc_soc.png", dpi=220)
    plot_voltage_results(t_rel, v_meas, v_pred_hist)
    plt.savefig(OUTPUT_DIR / "ekf_1rc_voltage.png", dpi=220)
    plot_residual(t_rel, resid_hist)
    plt.savefig(OUTPUT_DIR / "ekf_1rc_residual.png", dpi=220)
    plt.close("all")

if __name__ == "__main__":
    main()

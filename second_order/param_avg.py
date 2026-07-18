import numpy as np
import pandas as pd
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "second_order"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOC_GRID = np.linspace(0.0, 1.0, 201)
MAX_OCV_POLY_ORDER = 9

def find_optional(paths):
    for p in paths:
        path = OUTPUT_DIR / p
        if path.exists():
            return path
    return None

def eval_poly_from_npz(npz_obj, x, key="ocv_coeffs"):
    coeffs = np.array(npz_obj[key])
    if coeffs.size == 0:
        return None
    return np.polyval(coeffs, x)

def fit_poly_adaptive(x, y, max_order):
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    if len(x) < 2:
        return None, None
    order = min(max_order, len(x) - 1)
    coeffs = np.polyfit(x, y, order)
    return coeffs, order

def load_table(path, required_cols):
    if path is None or (not path.exists()):
        return None
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None
    if len(df) == 0:
        return None
    return df

def table_mean_or_nan(df, col):
    if df is None or col not in df.columns or len(df) == 0:
        return np.nan
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))

def average_available(*vals):
    vals = [float(v) for v in vals if np.isfinite(v)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))

def main():
    charge_2rc_table_path = find_optional(["identified_charge_params_table_2rc.csv"])
    discharge_2rc_table_path = find_optional(["identified_discharge_params_table_2rc.csv"])
    charge_1rc_table_path = find_optional(["identified_charge_params_table.csv"])
    discharge_1rc_table_path = find_optional(["identified_discharge_params_table.csv"])

    charge_2rc_fit_path = find_optional(["identified_charge_param_fits_2rc.npz"])
    discharge_2rc_fit_path = find_optional(["identified_discharge_param_fits_2rc.npz"])
    charge_1rc_fit_path = find_optional(["identified_charge_param_fits.npz"])
    discharge_1rc_fit_path = find_optional(["identified_discharge_param_fits.npz"])

    charge_2rc_df = load_table(charge_2rc_table_path, ["R0_ohm", "R1_ohm", "C1_F", "R2_ohm", "C2_F"])
    discharge_2rc_df = load_table(discharge_2rc_table_path, ["R0_ohm", "R1_ohm", "C1_F", "R2_ohm", "C2_F"])
    charge_1rc_df = load_table(charge_1rc_table_path, ["R0_ohm", "R1_ohm", "C_F"])
    discharge_1rc_df = load_table(discharge_1rc_table_path, ["R0_ohm", "R1_ohm", "C_F"])

    if charge_2rc_fit_path is not None:
        charge_fit = np.load(charge_2rc_fit_path, allow_pickle=True)
        charge_ocv_source = charge_2rc_fit_path.name
    elif charge_1rc_fit_path is not None:
        charge_fit = np.load(charge_1rc_fit_path, allow_pickle=True)
        charge_ocv_source = charge_1rc_fit_path.name
    else:
        charge_fit = None
        charge_ocv_source = ""

    if discharge_2rc_fit_path is not None:
        discharge_fit = np.load(discharge_2rc_fit_path, allow_pickle=True)
        discharge_ocv_source = discharge_2rc_fit_path.name
    elif discharge_1rc_fit_path is not None:
        discharge_fit = np.load(discharge_1rc_fit_path, allow_pickle=True)
        discharge_ocv_source = discharge_1rc_fit_path.name
    else:
        discharge_fit = None
        discharge_ocv_source = ""

    charge_ocv_grid = eval_poly_from_npz(charge_fit, SOC_GRID) if charge_fit is not None else None
    discharge_ocv_grid = eval_poly_from_npz(discharge_fit, SOC_GRID) if discharge_fit is not None else None

    if charge_ocv_grid is None and discharge_ocv_grid is None:
        raise FileNotFoundError("Could not find charge/discharge fit files with OCV coefficients.")

    if charge_ocv_grid is not None and discharge_ocv_grid is not None:
        ocv_avg_grid = 0.5 * (charge_ocv_grid + discharge_ocv_grid)
    elif charge_ocv_grid is not None:
        ocv_avg_grid = charge_ocv_grid.copy()
    else:
        ocv_avg_grid = discharge_ocv_grid.copy()

    ocv_coeffs, ocv_order = fit_poly_adaptive(SOC_GRID, ocv_avg_grid, MAX_OCV_POLY_ORDER)

    avg_r0 = average_available(
        table_mean_or_nan(charge_2rc_df, "R0_ohm"),
        table_mean_or_nan(discharge_2rc_df, "R0_ohm"),
        table_mean_or_nan(charge_1rc_df, "R0_ohm"),
        table_mean_or_nan(discharge_1rc_df, "R0_ohm"),
    )

    avg_r1 = average_available(
        table_mean_or_nan(charge_2rc_df, "R1_ohm"),
        table_mean_or_nan(discharge_2rc_df, "R1_ohm"),
    )
    avg_c1 = average_available(
        table_mean_or_nan(charge_2rc_df, "C1_F"),
        table_mean_or_nan(discharge_2rc_df, "C1_F"),
    )
    avg_r2 = average_available(
        table_mean_or_nan(charge_2rc_df, "R2_ohm"),
        table_mean_or_nan(discharge_2rc_df, "R2_ohm"),
    )
    avg_c2 = average_available(
        table_mean_or_nan(charge_2rc_df, "C2_F"),
        table_mean_or_nan(discharge_2rc_df, "C2_F"),
    )

    if not np.isfinite(avg_r1) or not np.isfinite(avg_c1) or not np.isfinite(avg_r2) or not np.isfinite(avg_c2):
        raise ValueError(
            "Could not compute valid averaged 2RC parameters. Run the 2RC identification scripts first."
        )

    avg_tau1 = avg_r1 * avg_c1
    avg_tau2 = avg_r2 * avg_c2

    qd_ah = np.nan
    qd_source = ""
    if discharge_2rc_fit_path is not None:
        fit = np.load(discharge_2rc_fit_path, allow_pickle=True)
        if "total_discharged_ah" in fit.files:
            qd_ah = float(fit["total_discharged_ah"])
            qd_source = discharge_2rc_fit_path.name
    if (not np.isfinite(qd_ah)) and discharge_1rc_fit_path is not None:
        fit = np.load(discharge_1rc_fit_path, allow_pickle=True)
        if "total_discharged_ah" in fit.files:
            qd_ah = float(fit["total_discharged_ah"])
            qd_source = discharge_1rc_fit_path.name
    if not np.isfinite(qd_ah):
        raise ValueError("Could not find total discharged capacity Qd in the discharge fit files.")
    qd_as = qd_ah * 3600.0

    summary = pd.DataFrame(
        {
            "parameter": [
                "avg_R0_ohm",
                "avg_R1_ohm",
                "avg_C1_F",
                "avg_tau1_s",
                "avg_R2_ohm",
                "avg_C2_F",
                "avg_tau2_s",
                "Qd_Ah",
                "Qd_As",
            ],
            "value": [
                avg_r0,
                avg_r1,
                avg_c1,
                avg_tau1,
                avg_r2,
                avg_c2,
                avg_tau2,
                qd_ah,
                qd_as,
            ],
        }
    )
    summary.to_csv(OUTPUT_DIR / "identified_params_avg_2rc.csv", index=False)

    np.savez(
        OUTPUT_DIR / "identified_param_avg_2rc.npz",
        avg_r0_ohm=avg_r0,
        avg_r1_ohm=avg_r1,
        avg_c1_f=avg_c1,
        avg_tau1_s=avg_tau1,
        avg_r2_ohm=avg_r2,
        avg_c2_f=avg_c2,
        avg_tau2_s=avg_tau2,
        qd_ah=qd_ah,
        qd_as=qd_as,
        ocv_coeffs=ocv_coeffs,
        ocv_order=-1 if ocv_order is None else ocv_order,
        soc_grid=SOC_GRID,
        ocv_charge_grid=np.array([]) if charge_ocv_grid is None else charge_ocv_grid,
        ocv_discharge_grid=np.array([]) if discharge_ocv_grid is None else discharge_ocv_grid,
        ocv_avg_grid=ocv_avg_grid,
        charge_ocv_source=np.array(charge_ocv_source),
        discharge_ocv_source=np.array(discharge_ocv_source),
        qd_source=np.array(qd_source),
    )

    print("Saved identified_params_avg_2rc.csv")
    print("Saved identified_param_avg_2rc.npz")
    print(f"avg_R0 = {avg_r0:.8f} ohm")
    print(f"avg_R1 = {avg_r1:.8f} ohm, avg_C1 = {avg_c1:.3f} F, avg_tau1 = {avg_tau1:.3f} s")
    print(f"avg_R2 = {avg_r2:.8f} ohm, avg_C2 = {avg_c2:.3f} F, avg_tau2 = {avg_tau2:.3f} s")
    print(f"Qd = {qd_ah:.6f} Ah")
    print(f"Charge OCV source: {charge_ocv_source or 'none'}")
    print(f"Discharge OCV source: {discharge_ocv_source or 'none'}")

if __name__ == "__main__":
    main()

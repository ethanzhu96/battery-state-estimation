import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "first_order"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISCHARGE_TABLE = OUTPUT_DIR / "identified_discharge_params_table.csv"
CHARGE_TABLE = OUTPUT_DIR / "identified_charge_params_table.csv"
DISCHARGE_FITS = OUTPUT_DIR / "identified_discharge_param_fits.npz"
CHARGE_FITS = OUTPUT_DIR / "identified_charge_param_fits.npz"

def main():
    dis_df = pd.read_csv(DISCHARGE_TABLE)
    chg_df = pd.read_csv(CHARGE_TABLE)
    dis_fit = np.load(DISCHARGE_FITS)
    chg_fit = np.load(CHARGE_FITS)

    z = np.linspace(0.0, 1.0, 400)
    ocv_dis = np.polyval(dis_fit["ocv_coeffs"], z)
    ocv_chg = np.polyval(chg_fit["ocv_coeffs"], z)

    plt.figure(figsize=(11, 7))
    plt.scatter(dis_df["SOC"], dis_df["OCV_V"], label="Discharge OCV points", s=36)
    plt.scatter(chg_df["SOC"], chg_df["OCV_V"], label="Charge OCV points", s=36, marker="s")
    plt.plot(z, ocv_dis, label=f"Discharge OCV fit (order {int(dis_fit['ocv_order'])})", linewidth=2)
    plt.plot(z, ocv_chg, label=f"Charge OCV fit (order {int(chg_fit['ocv_order'])})", linewidth=2)

    plt.xlabel("SOC")
    plt.ylabel("OCV (V)")
    plt.title("Charge + Discharge OCV")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / "charge_discharge_ocv_plot.png"
    plt.savefig(out_path, dpi=220)
    print(f"Saved: {out_path.name}")
    plt.close("all")

if __name__ == "__main__":
    main()

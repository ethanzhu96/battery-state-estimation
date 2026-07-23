import pandas as pd
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "first_order"

def load_table(name):
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing parameter table: {path}")
    return pd.read_csv(path)

def main():
    dis_df = load_table("identified_discharge_params_table.csv")
    chg_df = load_table("identified_charge_params_table.csv")

    dis_r0 = dis_df["R0_ohm"].mean()
    dis_r1 = dis_df["R1_ohm"].mean()
    dis_c1 = dis_df["C_F"].mean()

    chg_r0 = chg_df["R0_ohm"].mean()
    chg_r1 = chg_df["R1_ohm"].mean()
    chg_c1 = chg_df["C_F"].mean()

    print("Discharge means:")
    print("R0 =", dis_r0)
    print("R1 =", dis_r1)
    print("C1 =", dis_c1)

    print("\nCharge means:")
    print("R0 =", chg_r0)
    print("R1 =", chg_r1)
    print("C1 =", chg_c1)

    print("\nAveraged constants for EKF:")
    print("R0_avg =", 0.5 * (dis_r0 + chg_r0))
    print("R1_avg =", 0.5 * (dis_r1 + chg_r1))
    print("C1_avg =", 0.5 * (dis_c1 + chg_c1))

if __name__ == "__main__":
    main()

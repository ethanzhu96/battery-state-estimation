# Battery State Estimation

Code for lithium-ion battery state-of-charge estimation using first- and second-order RC equivalent-circuit models, parameter identification, and Extended Kalman Filtering.

This repository is code-only. Raw cell test data, generated parameter files, fitted `.npz` artifacts, and result plots are intentionally excluded.

## Structure

```text
battery-state-estimation/
├── first_order/          # 1RC parameter ID and EKF scripts
├── second_order/         # 2RC parameter ID and EKF scripts
├── data/                 # local raw inputs; ignored by git
│   ├── first_order/
│   └── second_order/
├── outputs/              # generated tables, fits, plots; ignored by git
│   ├── first_order/
│   ├── second_order/
│   └── comparison/
├── compare_1rc_2rc.py
├── requirements.txt
└── README.md
```

## Data convention

The scripts assume the dataset convention used in the original workflow:

- positive current = charging
- negative current = discharging
- required raw CSV columns generally include `Test Time (s)`, `Current (A)`, and `Voltage (V)`

The SOC-labeling scripts generate `udds_between_full_charges_with_soc.csv` and `udds_soc_summary.txt`. The parameter-identification scripts generate both fitted parameter tables and `.npz` polynomial-fit files. EKF scripts then consume the SOC-labeled UDDS file, summary file, and parameter-fit `.npz` files.

## Typical workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

Place raw CSV files locally under `data/first_order/` or `data/second_order/`. Generated files will be written to `outputs/first_order/` or `outputs/second_order/`.

Example commands:

```bash
python first_order/udds_soc_labeling.py
python first_order/parameter_identification_charge.py
python first_order/parameter_identification_discharge.py
python first_order/EKF_SoC.py

python second_order/make_udds_soc.py
python second_order/parameter_id_charge.py
python second_order/parameter_id_discharge.py
python second_order/EKF_SoC_RC2.py

python compare_1rc_2rc.py
```

Expected first-order generated inputs for the EKF:

```text
outputs/first_order/udds_between_full_charges_with_soc.csv
outputs/first_order/udds_soc_summary.txt
outputs/first_order/identified_charge_param_fits.npz
outputs/first_order/identified_discharge_param_fits.npz
```

Expected second-order generated inputs for the EKF:

```text
outputs/second_order/udds_between_full_charges_with_soc.csv
outputs/second_order/udds_soc_summary.txt
outputs/second_order/identified_charge_param_fits_2rc.npz
outputs/second_order/identified_discharge_param_fits_2rc.npz
```

## Notes

This project is intended as a research/prototype implementation, not a production BMS library.

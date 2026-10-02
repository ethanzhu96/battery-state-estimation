# Battery State Estimation

To run the 20-epoch SOH-only initial-charge ablation:

```bash
python -m machine_learning.train_soc_lstm --soh-only --initial-soc 0.5
```

This selects whole trajectories before splitting and normalization, leaving
current/voltage as the only model inputs. It uses 872 training windows and
436 windows each for validation and test with the current generated dataset.
Omit `--initial-soc` for the original all-initial-SOC comparison. The profile
split, model, window length, batch size, learning rate, seed, and 20 epochs
are unchanged. The smaller dataset has fewer optimizer updates per epoch;
this comparison does not match update counts. Metrics still use the final
epoch, as in the original training script.

## SOH memorization diagnostic

Run from the repository root:

```bash
python -m machine_learning.overfit_soh
```

This trains the existing LSTM with SOH-only MSE on four fixed 1,800-sample
windows at SOH 0.7, 0.8, 0.9, and 1.0. Each uses the square current profile,
initial SOC 0.5, and the first window of its trajectory. Normalization uses
only those four windows. Evaluation uses the same training examples: this
checks memorization, not generalization. The constant-mean baseline has
10 percentage points MAE and approximately 11.18 percentage points RMSE.

The run stops below 1 percentage point training MAE or after 2,000 updates.
`outputs/soh_overfit/` receives the best checkpoint, predictions, per-step
history, and a JSON summary with configuration and normalization statistics.
Use `--output` to preserve separate runs; `--steps`, `--sequence-length`,
`--seed`, and `--learning-rate` are configurable. Existing full-data training
is unchanged.

Code for lithium-ion battery state-of-charge estimation using first- and second-order RC equivalent-circuit models, parameter identification, and Extended Kalman Filtering.

Raw cell test data is included so a fresh clone has the inputs needed to run the project. Generated parameter files, fitted `.npz` artifacts, and result plots under `outputs/` are intentionally excluded.

## Structure

```text
battery-state-estimation/
├── first_order/          # 1RC parameter ID and EKF scripts
├── second_order/         # 2RC parameter ID and EKF scripts
├── data/                 # raw inputs; tracked by git
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

Raw CSV files live under `data/first_order/` or `data/second_order/` and are tracked by Git. Generated files will be written to `outputs/first_order/` or `outputs/second_order/`.

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

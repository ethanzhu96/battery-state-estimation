# Battery State Estimation

## Static MLP versus vanilla real-data LSTM

Run the small MLP search on the exact Cell 9 aging split used by the vanilla
LSTM, with a direct comparison to its saved checkpoint:

```powershell
.\machine_learning\train_cell9_aging.ps1 -Model mlp -MlpSearch -Output outputs/mlp_cell9_0_200 -CompareLstmCheckpoint outputs/real_cell9_0_200/best.pt
```

The existing real-data LSTM predicts one capacity-derived SOH label per
nonoverlapping 1,800-second current/voltage window, not a sequence of previous
aging cycles. The MLP receives **only the current and voltage at that window's
last time step**, as a two-element static vector. It receives no flattened
history, cycle number, capacity, temperature, or extracted pulse features.
This explicitly tests endpoint information versus 30 minutes of history.

Loading, capacity labels, raw-time 80/20 train/validation boundaries, window
creation, sample order, and normalization are shared with `train_real_soh`.
The same mean/std are fitted across training-window measurements and applied
to both models before MLP endpoint selection; no endpoint-only refitting.
The comparison checks checkpoint data sources, labels, split, sample counts,
and normalization for compatibility. It leaves the LSTM architecture and
matched-pulse ablation experiment unchanged.

Six predefined MLP candidates explore layer count 1/2/3, width 32/64/128,
dropout 0/0.1, and Adam learning rates 0.0003/0.001/0.003. Activation defaults
to ReLU. Each trains for the same configured maximum epochs; selection is
the lowest validation RMSE across configurations and epochs. The existing
trainer has no early stopping, so none is introduced. The historical LSTM
checkpoint was selected by validation MAE; this difference is reported.

For a single configuration, use the shared `train_real_soh` command with
`--model mlp` and its existing `--input`, `--capacity-csv`, and
`--bol-capacity-csv` flags. Configure `--hidden-layers`, `--hidden-dim`,
`--dropout`, `--activation relu|tanh|gelu`, and `--learning-rate`.
`--mlp-search` uses the six predefined candidates instead of those dimensions.

Reports include RMSE/MAE in SOH percentage points, R² (undefined when all labels
are equal), selected configuration/epoch, and training-mean predictions.
`metrics.json`, `search.csv`, `history.csv`, `best.pt`, and comparison predictions
are saved under the chosen output directory.

**There is no test set in the current real-data 80/20 pipeline.** Test metrics
are explicitly unavailable rather than reusing validation. To supply a test
set later, use matching `--test-input` and `--test-capacity-csv` lists in the
shared trainer; it evaluates identical test windows for both models only
after MLP selection and uses frozen training statistics. Files overlapping
training/validation are rejected. The current single-cell comparison remains
exploratory and cannot establish unseen-cell performance.

## Matched pulse comparison

Run the controlled voltage/pooling/feature ablations with:

```powershell
py -3.13 -u -m machine_learning.compare_pulse_soh --ablations --output outputs/pulse_soh_ablations
```

This runs all eight combinations of absolute/relative voltage, mean/last-state
pooling, and absence/presence of the four pulse features. Relative voltage
subtracts each event's pre-pulse resting voltage. Hybrid features concatenate
to the pooled encoder output. All variants use the same paired folds, seed per
fold, optimizer, and fixed update budget. Sequence and feature normalization
use training folds only. The original LSTM baseline and regression controls
are rerun. Reports include RMSE as well as MAE to expose large individual errors.

Test and experiment source code is intended for GitHub. Generated contents of
`outputs/` are ignored, except `.gitkeep` files. The private retrospective
research notebook at `research/research_log.ipynb` is also ignored.

```powershell
py -3.13 -u -m machine_learning.compare_pulse_soh
```

Uses the supplied Cell 9 0/200-cycle 25 C filenames in Downloads (override
`--data-dir`). Extracts 10-second negative pulses with 5 seconds of prior
rest and 140 seconds of subsequent rest, excluding gaps over 2 seconds.
Pairs events one-to-one by resting voltage (within 40 mV), pulse current
(within 0.05 A), and duration (within 1 second). Resting voltage approximates
charge region; it is not exact SOC matching. Capacity files supply labels
only and must use matching test rates and voltage cutoffs.

Compares a mean predictor, resting-voltage-only ridge regression, pulse-feature
ridge regression, pulse-plus-voltage ridge regression, and the existing LSTM
on exactly the same events. Five folds keep each matched fresh/aged pair in
the same fold; folds are interleaved by resting-voltage order. Normalization
uses training folds only. Ridge alpha is fixed at 1; the LSTM uses 200 fixed
full-batch updates per fold (`--epochs`), without held-out epoch selection.
This measures exploratory within-cell checkpoint discrimination, not unseen
cell or unseen checkpoint performance. Repeated events are correlated.

`outputs/pulse_soh_comparison` contains matched features, out-of-fold predictions,
metrics, source paths, training history, fold checkpoints, and two plots.
Pulse resistance estimates include transient/sampling effects. Relaxation
features are voltage recovery from 1 to 30/120 seconds after pulse removal.
Check the adapter and extraction with:

```powershell
py -3.13 -m unittest machine_learning.test_pulse_soh machine_learning.test_real_soh
```

## Train on the available real files

For aging data, `train_real_soh` also accepts `--input` (one or more incremental
CSVs), `--capacity-csv` (one matching capacity file per input, in the same
order), and `--bol-capacity-csv` (a fresh capacity test for the same cell).
Capacity labels are computed from measured discharge capacity; discharge
currents must match within 5% and cutoff voltages within 0.02 V. Use the same
cell, temperature, and capacity-test protocol. The loader splits the first
80% and last 20% of each continuous trace before creating nonoverlapping
windows, then fits normalization only on the training windows. This is a
same-cell convergence experiment, not held-out-cell or held-out-aging-stage
evaluation. One checkpoint still supplies just one health label.

```powershell
py -3.13 -u -m machine_learning.train_real_soh --epochs 100
```

This initializes a fresh LSTM and trains SOH-only on Cell 9's real 1C
incremental file, validating on the same cell's 0.1C incremental file.
Files default to the user's Downloads folder; override with `--data-dir`.
These specific inputs must be the supplied 0-cycle files: every target is
100% SOH, so this is only a convergence diagnostic. A constant 100% predictor
already has zero error; low loss does not demonstrate aging estimation or
unseen-cell generalization. Do not substitute aged files into this command.

Normalization is fitted only on training windows. Measurements are resampled
to 1 second, with nonoverlapping 1,800-sample windows and gaps over 65 seconds
split. Each epoch prints MSE and training/validation MAE in percentage points.
`outputs/real_soh_training` contains history, label/source metadata, the best
validation checkpoint, and its predictions. Override `--output` to preserve
separate experiments. This command does not load synthetic-trained weights.

## Real-data SOH evaluation

Save a full synthetic training checkpoint selected by validation SOH MAE:

```powershell
py -3.13 -m machine_learning.train_soc_lstm --soh-only --initial-soc 0.5 --output outputs/soh_transfer_synthetic
```

When `--output` is supplied, training saves `best.pt` with model configuration,
training normalization, sampling interval, window length, seed, and split IDs.
It also saves epoch history and synthetic test predictions/metrics, evaluating
the best validation epoch rather than the final epoch. Without `--output`,
the original final-epoch behavior is preserved.

Evaluate the available fresh Cell 9 files (adjust the download directory):

```powershell
py -3.13 -m machine_learning.evaluate_real_soh `
  --checkpoint outputs/soh_transfer_synthetic/best.pt `
  --input "$env:USERPROFILE/Downloads/SAMSUNG_Cell_9_incremental_1C_Channel_3_Wb_1.CSV" "$env:USERPROFILE/Downloads/SAMSUNG_Cell_9_incremental_01C_Channel_3_Wb_1.CSV" `
  --capacity-csv "$env:USERPROFILE/Downloads/SAMSUNG_Cell_9_capacity_02C_Channel_3_Wb_1.CSV" `
  --bol-capacity-csv "$env:USERPROFILE/Downloads/SAMSUNG_Cell_9_capacity_02C_Channel_3_Wb_1.CSV" `
  --cell-id SAMSUNG_Cell_9 --cycle 0 --temperature-c 25 `
  --output outputs/real_cell9_cycle0
```

This is synthetic-to-real inference, not training on real data. Inputs are
current and voltage only, with positive charging current. The adapter uses
previous-sample hold for current and linear interpolation for voltage at the
checkpoint sampling interval. Gaps above `--max-gap-s` (default 65 seconds)
split the trace; smaller gaps, including sparse rest logging, are resampled.
Timestamp resets, missing measurements, and discharge-counter resets are
rejected. Default windows are nonoverlapping (stride 1,800 samples); incomplete
tails are excluded. No window crosses a file or gap boundary.

SOH is the capacity-test discharge-counter increment divided by the matching
beginning-of-life capacity-test increment, each taken over its longest
continuous discharge segment. Verify matching cell identity, test rate, cutoff,
and temperature yourself before using other files. For these fresh-cell files,
the same capacity test is numerator and denominator: SOH is 100% by definition.
One fresh cell cannot establish SOH accuracy across aging; a constant 100%
predictor has zero error here. Temperature/cycle CLI arguments record supplied
metadata, not measured verification. Predictions are not clipped to [0, 1].

`predictions.csv` and `metrics.json` include per-window predictions, per-file
MAE/RMSE/bias, source paths, label provenance, and preprocessing assumptions.
Training normalization is reused unchanged; capacity and metadata are never
model features. Run adapter checks with:

```powershell
py -3.13 -m unittest machine_learning.test_real_soh
```

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

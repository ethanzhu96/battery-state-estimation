"""Evaluate a synthetic-trained SOH checkpoint on Samsung CSV measurements.

No fitting or normalization adaptation is performed on the real measurements.
Capacity labels require comparable capacity tests for the same cell.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from machine_learning.models.lstm import LSTM


TIME = 'Test Time (s)'
CURRENT = 'Current (A)'
VOLTAGE = 'Voltage (V)'


def read_measurements(path):
    data = pd.read_csv(path)
    required = [TIME, CURRENT, VOLTAGE]
    if not set(required).issubset(data.columns):
        raise ValueError(f'{path}: missing time, current, or voltage columns')
    if len(data) < 2 or not np.isfinite(data[required].to_numpy(dtype=float)).all():
        raise ValueError(f'{path}: insufficient rows or nonfinite measurements')
    if (np.diff(data[TIME]) <= 0).any():
        raise ValueError(f'{path}: timestamps must strictly increase; separate/reset tests first')
    return data


def discharge_capacity(path):
    """Use the counter increment over the longest continuous discharge segment."""
    data = read_measurements(path)
    column = 'Discharge Capacity (Ah)'
    if column not in data:
        raise ValueError(f'{path}: missing {column}')
    active = data[CURRENT].to_numpy() < -0.05
    edges = np.diff(np.r_[False, active, False].astype(int))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    if not runs:
        raise ValueError(f'{path}: no discharge segment (expected negative discharge current)')
    t = data[TIME].to_numpy()
    start, end = max(runs, key=lambda run: t[run[1]-1] - t[run[0]])
    q = data[column].to_numpy(dtype=float)
    counters = q[max(0, start-1):end]
    if not np.isfinite(counters).all() or (np.diff(counters) < -1e-6).any():
        raise ValueError('Discharge capacity counter is missing or resets during the segment')
    capacity = float(q[end-1] - q[max(0, start-1)])
    if capacity <= 0:
        raise ValueError('Nonpositive discharge capacity')
    return capacity, {'start_s': float(t[start]), 'end_s': float(t[end-1]),
                      'end_voltage_v': float(data[VOLTAGE].iloc[end-1])}


def resample_segments(data, dt, max_gap):
    """Preserve current steps with previous-sample hold; interpolate voltage.

Never extrapolate or bridge gaps larger than max_gap. Sparse logged rests
up to max_gap are interpolated; this is an explicit evaluation assumption.
"""
    t = data[TIME].to_numpy(dtype=float)
    boundaries = np.r_[0, np.flatnonzero(np.diff(t) > max_gap) + 1, len(t)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right - left < 2:
            continue
        times = t[left:right]
        grid = np.arange(times[0], times[-1], dt)
        indices = np.searchsorted(times, grid, side='right') - 1
        current = data[CURRENT].to_numpy()[left:right][indices]
        voltage = np.interp(grid, times, data[VOLTAGE].to_numpy()[left:right])
        yield grid, np.column_stack([current, voltage])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path, nargs='+')
    parser.add_argument('--capacity-csv', required=True, type=Path)
    parser.add_argument('--bol-capacity-csv', required=True, type=Path)
    parser.add_argument('--cell-id', required=True)
    parser.add_argument('--cycle', required=True, type=int)
    parser.add_argument('--temperature-c', required=True, type=float)
    parser.add_argument('--stride', type=int, default=1800,
                        help='Window stride in samples; default nonoverlapping 30-minute windows')
    parser.add_argument('--max-gap-s', type=float, default=65,
                        help='Allow sparse rest logging up to this gap; split larger gaps')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stride < 1 or args.max_gap_s <= 0:
        parser.error('stride and max-gap-s must be positive')
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('feature_columns') != ['current_a', 'voltage_v']:
        raise ValueError('Checkpoint must declare current/voltage feature order')
    if checkpoint.get('current_convention') != 'positive_charge':
        raise ValueError('Checkpoint current convention incompatible with Samsung CSVs')
    length = int(checkpoint['sequence_length'])
    dt = float(checkpoint['sample_interval_s'])
    mean, std = np.asarray(checkpoint['feature_mean']), np.asarray(checkpoint['feature_std'])
    if length < 1 or dt <= 0 or mean.shape != (2,) or std.shape != (2,) or not np.isfinite([mean, std]).all() or (std <= 0).any():
        raise ValueError('Invalid checkpoint preprocessing metadata')
    model = LSTM(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    capacity, capacity_segment = discharge_capacity(args.capacity_csv)
    bol_capacity, _ = discharge_capacity(args.bol_capacity_csv)
    target = capacity / bol_capacity
    rows, summaries = [], []
    for path in args.input:
        data = read_measurements(path)
        file_rows = []
        for segment_id, (times, features) in enumerate(resample_segments(data, dt, args.max_gap_s)):
            for start in range(0, len(times) - length + 1, args.stride):
                window = features[start:start+length]
                x = torch.tensor((window - mean) / std, dtype=torch.float32).unsqueeze(0)
                with torch.inference_mode():
                    prediction = float(model(x)[1].item())
                if not np.isfinite(prediction):
                    raise ValueError('Nonfinite prediction')
                file_rows.append({'source': str(path.resolve()), 'segment': segment_id,
                                  'start_s': float(times[start]), 'end_s': float(times[start+length-1]),
                                  'true_soh': target, 'predicted_soh': prediction,
                                  'error_pp': 100 * (prediction-target)})
        if not file_rows:
            raise ValueError(f'{path}: no complete {length}-sample windows after gap splitting')
        errors = np.array([row['error_pp'] for row in file_rows])
        result = {'source': str(path.resolve()), 'windows': len(file_rows),
                  'mae_pp': float(np.abs(errors).mean()),
                  'rmse_pp': float(np.sqrt(np.mean(errors**2))), 'bias_pp': float(errors.mean()),
                  'mean_prediction_percent': float(100*target+errors.mean()),
                  'raw_rows': len(data), 'max_raw_gap_s': float(np.diff(data[TIME]).max())}
        summaries.append(result)
        rows.extend(file_rows)
        print(f'{path.name}: {len(file_rows)} windows; MAE {result["mae_pp"]:.3f} pp; '
              f'mean prediction {result["mean_prediction_percent"]:.2f}%', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / 'predictions.csv', index=False)
    report = {'evaluation': 'synthetic-to-real transfer, one cell and one aging checkpoint',
              'limitations': ['Cannot establish accuracy across aging or unseen real cells.',
                              'Windows from one cell are correlated; not independent batteries.',
                              'No real-data fitting, clipping, or normalization adaptation.',
                              'Capacity protocols and cell identity must match across label files.'],
              'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_epoch': checkpoint['epoch'],
              'cell_id': args.cell_id, 'cycle': args.cycle, 'temperature_c': args.temperature_c,
              'capacity_ah': capacity, 'bol_capacity_ah': bol_capacity, 'true_soh': target,
              'capacity_csv': str(args.capacity_csv.resolve()),
              'bol_capacity_csv': str(args.bol_capacity_csv.resolve()),
              'capacity_segment': capacity_segment,
              'same_capacity_reference': args.capacity_csv.resolve() == args.bol_capacity_csv.resolve(),
              'sequence_length': length, 'sample_interval_s': dt, 'stride': args.stride,
              'max_gap_s': args.max_gap_s,
              'resampling': 'previous-sample hold current; linear voltage; split gaps above max_gap_s',
              'feature_mean': mean.tolist(), 'feature_std': std.tolist(),
              'constant_85_percent_baseline_mae_pp': 100*abs(.85-target),
              'constant_100_percent_baseline_mae_pp': 100*abs(1-target), 'files': summaries}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Capacity {capacity:.6f} Ah; reference {bol_capacity:.6f} Ah; SOH {100*target:.2f}%')
    print('Single-checkpoint transfer diagnostic only. Results saved to', args.output)


if __name__ == '__main__':
    main()

"""Matched pulse diagnostic on Cell 9 at 0/200 cycles, 25 C.

This tests within-cell checkpoint discrimination, not unseen-cell SOH accuracy.
Capacity is used exclusively to construct labels. Matching uses pre-pulse
resting voltage as an observable proxy for charge region, not measured SOC.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch

from machine_learning.evaluate_real_soh import CURRENT, TIME, VOLTAGE, read_measurements
from machine_learning.train_real_soh import comparable_target
from machine_learning.models.lstm import LSTM


FEATURES = ['r_on_ohm', 'r_off_ohm', 'recovery_120_v', 'relaxation_30_v']


class PulseLSTM(LSTM):
    """Existing encoder with explicit pooling and optional measured features."""
    def __init__(self, pooling='mean', feature_count=0):
        super().__init__(input_size=2, hidden_size=64, num_layers=1)
        if pooling not in ('mean', 'last'):
            raise ValueError('Unknown pooling')
        self.pooling = pooling
        self.feature_count = feature_count
        if feature_count:
            self.soh_head = torch.nn.Linear(64+feature_count, 1)

    def forward(self, x, features=None):
        encoded, _ = self.lstm(x)
        pooled = encoded.mean(dim=1) if self.pooling == 'mean' else encoded[:, -1]
        if self.feature_count:
            if features is None or features.shape != (len(x), self.feature_count):
                raise ValueError('Expected one pulse-feature vector per sequence')
            pooled = torch.cat([pooled, features], dim=1)
        return self.soc_head(encoded), self.soh_head(pooled)


def relative_voltage(sequences, pre_voltage):
    result = sequences.copy()
    result[:, :, 1] -= np.asarray(pre_voltage)[:, None]
    return result


def extract_pulses(data):
    t, current, voltage = [data[c].to_numpy(dtype=float) for c in [TIME, CURRENT, VOLTAGE]]
    active = current < -1.0
    edges = np.diff(np.r_[False, active, False].astype(int))
    pulses, sequences = [], []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if start == 0 or end == len(t):
            continue
        onset, offset = t[start], t[end]
        duration = offset - onset
        if not 8 <= duration <= 12:
            continue
        # Require the entire observation to be a rest/pulse/rest event.
        before = (t >= onset-5) & (t < onset)
        after = (t >= offset) & (t <= onset+150)
        if not before.any() or not after.any() or t[-1] < onset+150:
            continue
        if np.max(np.abs(current[before])) > .01 or np.max(np.abs(current[after])) > .01:
            continue
        region = (t >= onset-6) & (t <= onset+151)
        if np.max(np.diff(t[region])) > 2:
            continue
        amplitude = float(np.median(current[start:end]))
        rest_v = float(np.median(voltage[before]))
        rest_t = t[end:]
        rest_voltage = voltage[end:]
        v1, v30, v120 = np.interp(offset+np.array([1., 30., 120.]), rest_t, rest_voltage)
        delta_current_on = current[start] - current[start-1]
        delta_current_off = current[end] - current[end-1]
        if abs(delta_current_on) < 1 or abs(delta_current_off) < 1:
            continue
        grid = onset + np.arange(-5., 151.)
        idx = np.searchsorted(t, grid, side='right')-1
        sequence = np.column_stack([current[idx], np.interp(grid, t, voltage)])
        pulse = {
            'onset_s': onset, 'offset_s': offset, 'duration_s': duration,
            'current_a': amplitude, 'pre_voltage_v': rest_v,
            'r_on_ohm': (voltage[start]-voltage[start-1])/delta_current_on,
            'r_off_ohm': (voltage[end]-voltage[end-1])/delta_current_off,
            'recovery_120_v': float(v120-v1), 'relaxation_30_v': float(v30-v1),
            'temperature_c': float(data.loc[region, 'Aux_Temperature_1 (C)'].mean())
                if 'Aux_Temperature_1 (C)' in data else None,
        }
        pulses.append(pulse)
        sequences.append(sequence)
    if not pulses:
        raise ValueError('No complete 10-second discharge pulse/rest events found')
    return pd.DataFrame(pulses), np.stack(sequences)


def match_pulses(fresh, aged, tolerance):
    voltage_distance = abs(fresh.pre_voltage_v.to_numpy()[:, None]-aged.pre_voltage_v.to_numpy()[None, :])
    current_distance = abs(fresh.current_a.to_numpy()[:, None]-aged.current_a.to_numpy()[None, :])
    duration_distance = abs(fresh.duration_s.to_numpy()[:, None]-aged.duration_s.to_numpy()[None, :])
    valid = (voltage_distance <= tolerance) & (current_distance <= .05) & (duration_distance <= 1.)
    cost = np.where(valid, voltage_distance, 1e6)
    left, right = linear_sum_assignment(cost)
    pairs = [(int(a), int(b)) for a, b in zip(left, right) if valid[a, b]]
    return sorted(pairs, key=lambda pair: fresh.pre_voltage_v.iloc[pair[0]], reverse=True)


def ridge_predict(train, target, test):
    mean, std = train.mean(axis=0), train.std(axis=0)
    std = np.where(std > 1e-12, std, 1.)
    train_x = np.column_stack([np.ones(len(train)), (train-mean)/std])
    test_x = np.column_stack([np.ones(len(test)), (test-mean)/std])
    penalty = np.eye(train_x.shape[1])
    penalty[0, 0] = 0
    beta = np.linalg.solve(train_x.T@train_x + penalty, train_x.T@target)
    return test_x@beta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path.home()/'Downloads')
    parser.add_argument('--output', type=Path, default=Path('outputs/pulse_soh_comparison'))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--voltage-tolerance', type=float, default=.04)
    parser.add_argument('--ablations', action='store_true',
                        help='Run voltage representation x pooling x pulse-feature factorial ablations')
    args = parser.parse_args()
    if args.epochs < 1 or args.voltage_tolerance <= 0:
        parser.error('epochs and voltage-tolerance must be positive')
    torch.set_num_threads(2)
    files = [args.data_dir/'SAMSUNG_Cell_9_incremental_1C_Channel_3_Wb_1.CSV',
             args.data_dir/'SAMSUNG_Cell_9_incremental_1C_Channel_3_Wb_1(1).CSV']
    caps = [args.data_dir/'SAMSUNG_Cell_9_capacity_01C_Channel_3_Wb_1(1).CSV',
            args.data_dir/'SAMSUNG_Cell_9_capacity_01C_Channel_3_Wb_1.CSV']
    aged_soh, aged_capacity, fresh_capacity = comparable_target(caps[1], caps[0])
    extracted = [extract_pulses(read_measurements(path)) for path in files]
    pairs = match_pulses(extracted[0][0], extracted[1][0], args.voltage_tolerance)
    if len(pairs) < 5:
        raise ValueError(f'Only {len(pairs)} matched pairs; need at least 5 for grouped cross-validation')
    frames, windows = [], []
    for pair_id, (a, b) in enumerate(pairs):
        for checkpoint, index in [(0, a), (200, b)]:
            source_index = int(checkpoint == 200)
            frame, seq = extracted[source_index]
            row = frame.iloc[index].to_dict()
            row.update(pair_id=pair_id, cycle=checkpoint, true_soh=1. if checkpoint == 0 else aged_soh)
            frames.append(row)
            windows.append(seq[index])
    events = pd.DataFrame(frames)
    seq = np.stack(windows)
    target = events.true_soh.to_numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output/'matched_features.csv', index=False)
    print(f'Extracted {len(extracted[0][0])} fresh / {len(extracted[1][0])} aged pulses; '
          f'{len(pairs)} matched pairs', flush=True)
    print(f'Labels: 100% / {100*aged_soh:.3f}%; same-cell diagnostic, 5 grouped folds', flush=True)
    variants = {'lstm': {'relative': False, 'pooling': 'mean', 'features': False}}
    if args.ablations:
        for relative in [False, True]:
            for pooling in ['mean', 'last']:
                for features in [False, True]:
                    if not relative and pooling == 'mean' and not features:
                        continue
                    name = f'lstm_{"relative" if relative else "absolute"}_{pooling}'
                    if features:
                        name += '_features'
                    variants[name] = {'relative': relative, 'pooling': pooling, 'features': features}
    outputs = {name: np.zeros(len(events)) for name in
               ['mean', 'voltage_only', 'pulse_features', 'pulse_plus_voltage', *variants]}
    history = []
    for fold in range(5):
        test = events.pair_id.to_numpy() % 5 == fold
        train = ~test
        outputs['mean'][test] = target[train].mean()
        for name, columns in [('voltage_only', ['pre_voltage_v']), ('pulse_features', FEATURES),
                              ('pulse_plus_voltage', FEATURES+['pre_voltage_v'])]:
            matrix = events[columns].to_numpy()
            outputs[name][test] = ridge_predict(matrix[train], target[train], matrix[test])
        for name, config in variants.items():
            torch.manual_seed(42+fold)
            representation = relative_voltage(seq, events.pre_voltage_v) if config['relative'] else seq
            mean, std = representation[train].mean(axis=(0, 1)), representation[train].std(axis=(0, 1))
            if (std == 0).any():
                raise ValueError('Zero training feature standard deviation')
            x = torch.tensor((representation-mean)/std, dtype=torch.float32)
            feature_matrix = events[FEATURES].to_numpy()
            feature_mean, feature_std = feature_matrix[train].mean(axis=0), feature_matrix[train].std(axis=0)
            feature_std = np.where(feature_std > 1e-12, feature_std, 1.)
            pulse_features = torch.tensor((feature_matrix-feature_mean)/feature_std, dtype=torch.float32)
            y = torch.tensor(target[train], dtype=torch.float32).unsqueeze(1)
            model = PulseLSTM(config['pooling'], len(FEATURES) if config['features'] else 0)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            # Fixed training budget; held-out fold never selects epochs or tuning.
            for epoch in range(1, args.epochs+1):
                model.train()
                optimizer.zero_grad()
                loss = (model(x[train], pulse_features[train])[1]-y).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite LSTM training loss')
                loss.backward()
                optimizer.step()
                history.append({'model': name, 'fold': fold, 'epoch': epoch, 'training_mse': loss.item()})
            model.eval()
            with torch.inference_mode():
                outputs[name][test] = model(x[test], pulse_features[test])[1].flatten().numpy()
            torch.save({'model_state_dict': model.state_dict(), 'feature_mean': mean.tolist(),
                        'feature_std': std.tolist(), 'pulse_feature_mean': feature_mean.tolist(),
                        'pulse_feature_std': feature_std.tolist(), 'pulse_feature_columns': FEATURES,
                        'variant': config, 'fold': fold, 'epochs': args.epochs},
                       args.output/f'{name}_fold_{fold}.pt')
            print(f'Fold {fold+1}/5 {name}: complete', flush=True)
        print(f'Fold {fold+1}/5 complete: {train.sum()} train / {test.sum()} held-out events', flush=True)
    metrics = {}
    for name, predictions in outputs.items():
        errors = 100*(predictions-target)
        metrics[name] = {'mae_pp': float(abs(errors).mean()),
                         'rmse_pp': float(np.sqrt(np.mean(errors**2))),
                         'bias_pp': float(errors.mean())}
        events[f'{name}_prediction'] = predictions
        print(f'{name}: MAE {metrics[name]["mae_pp"]:.3f} pp; RMSE {metrics[name]["rmse_pp"]:.3f} pp', flush=True)
    events.to_csv(args.output/'predictions.csv', index=False)
    pd.DataFrame(history).to_csv(args.output/'lstm_history.csv', index=False)
    differences = {}
    for column in FEATURES:
        fresh_values = events.loc[events.cycle == 0, column].to_numpy()
        aged_values = events.loc[events.cycle == 200, column].to_numpy()
        delta = aged_values-fresh_values
        differences[column] = {'fresh_mean': float(fresh_values.mean()), 'aged_mean': float(aged_values.mean()),
                               'paired_mean_change': float(delta.mean()),
                               'paired_change_std': float(delta.std(ddof=1)),
                               'positive_change_fraction': float((delta > 0).mean())}
    summary = {'fresh_capacity_ah': fresh_capacity, 'aged_capacity_ah': aged_capacity,
               'aged_soh': aged_soh, 'pairs': len(pairs), 'metrics': metrics,
               'paired_feature_changes': differences,
               'matching': '10-second negative pulses; amplitude within 0.05 A; duration within 1 s; '
                           f'pre-rest voltage within {args.voltage_tolerance} V; one-to-one',
               'sequence': '156 one-second samples, -5 to +150 seconds from pulse onset',
               'split': '5-fold by matched pair, interleaved in resting-voltage order; both ages kept together',
               'regression': 'ridge alpha=1; features standardized on training fold only',
               'lstm': f'64-unit encoder; {args.epochs} fixed updates per fold; seed 42+fold',
               'variants': variants,
               'limitations': ['One cell, two checkpoints, correlated repeated events; no independent cell replication.',
                               'Rest voltage is only a proxy for charge region; not exact SOC matching.',
                               'Checkpoint differences may include temperature, history, or measurement drift.',
                               'Resistance from sampled voltage jumps includes sampling/transient effects.',
                               'Grouped results are exploratory; no untouched final test set.'],
               'sources': [str(p.resolve()) for p in files+caps]}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, column in zip(axes.flat, FEATURES):
        for cycle, color in [(0, 'tab:blue'), (200, 'tab:orange')]:
            subset = events[events.cycle == cycle]
            ax.plot(subset.pre_voltage_v, subset[column], 'o-', label=f'{cycle} cycles', color=color)
        ax.set_xlabel('Pre-pulse resting voltage (V)')
        ax.set_ylabel(column)
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle('Matched discharge pulses — one cell, two aging checkpoints')
    fig.tight_layout()
    fig.savefig(args.output/'feature_comparison.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for index, (name, predictions) in enumerate(outputs.items()):
        ax.scatter(100*target+(index-2)*.12, 100*predictions, label=name, alpha=.6)
    ax.plot([94, 101], [94, 101], 'k--')
    ax.set(xlabel='Measured SOH (%) — jitter for visibility', ylabel='Held-out prediction (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output/'prediction_comparison.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    names = sorted(metrics, key=lambda name: metrics[name]['mae_pp'])
    ax.barh(names, [metrics[name]['mae_pp'] for name in names])
    ax.axvline(metrics['mean']['mae_pp'], color='red', linestyle='--', label='Mean baseline')
    ax.set_xlabel('Out-of-fold SOH MAE (percentage points)')
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output/'ablation_comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()

"""Memorization diagnostic: train and evaluate on the same four SOH windows."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from machine_learning.models.lstm import LSTM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--sequence-length', type=int, default=1800)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--output', type=Path, default=Path('outputs/soh_overfit'))
    args = parser.parse_args()
    if args.steps < 1 or args.sequence_length < 1:
        parser.error('steps and sequence-length must be positive')
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source = Path(__file__).parent / 'simulation' / 'clean_trajectories.csv'
    data = pd.read_csv(source)
    selected = data[(data.profile_id == 'square') & np.isclose(data.initial_soc, 0.5)]
    windows, labels, ids = [], [], []
    for soh in [0.7, 0.8, 0.9, 1.0]:
        candidates = selected[np.isclose(selected.soh, soh)]
        if candidates.trajectory_id.nunique() != 1:
            raise ValueError(f'Expected one square / initial SOC 0.5 trajectory for SOH {soh}')
        window = candidates.sort_values('time_s').iloc[:args.sequence_length]
        if len(window) != args.sequence_length:
            raise ValueError('Trajectory shorter than requested window')
        windows.append(window[['current_a', 'voltage_v']].to_numpy())
        labels.append(soh)
        ids.append(int(window.trajectory_id.iloc[0]))
    features = np.stack(windows)
    mean = features.mean(axis=(0, 1))
    std = features.std(axis=(0, 1))
    if not np.isfinite(features).all() or (std == 0).any():
        raise ValueError('Features must be finite with nonzero standard deviations')
    x = torch.tensor((features - mean) / std, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)
    model = LSTM(input_size=2, hidden_size=64, num_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best_mae = float('inf')
    print(f'Device: {device}; four windows of {args.sequence_length} samples', flush=True)
    print('TRAINING-SET MEMORIZATION ONLY. Mean baseline: MAE 10.000 pp, RMSE 11.180 pp.', flush=True)
    for step in range(args.steps + 1):
        model.eval()
        with torch.no_grad():
            predictions = model(x)[1]
            errors = predictions - y
            mae = 100 * errors.abs().mean().item()
            rmse = 100 * errors.square().mean().sqrt().item()
            values = predictions.flatten().cpu().tolist()
        history.append({'step': step, 'mae_pp': mae, 'rmse_pp': rmse,
                        **{f'prediction_{label:.1f}': value for label, value in zip(labels, values)}})
        if mae < best_mae:
            best_mae = mae
            best_step = step
            best_values = values
            best_rmse = rmse
            torch.save({'model_state_dict': model.state_dict(),
                        'feature_mean': mean.tolist(), 'feature_std': std.tolist(),
                        'step': step, 'seed': args.seed}, args.output / 'best.pt')
        if step % 100 == 0 or mae < 1.0 or step == args.steps:
            print(f'Step {step}: MAE {mae:.4f} pp; RMSE {rmse:.4f} pp; '
                  f'predictions {[round(100*v, 2) for v in values]}', flush=True)
        if mae < 1.0 or step == args.steps:
            break
        model.train()
        optimizer.zero_grad()
        loss = (model(x)[1] - y).square().mean()
        loss.backward()
        if not torch.isfinite(loss) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise RuntimeError('Nonfinite loss or gradients')
        optimizer.step()
    pd.DataFrame(history).to_csv(args.output / 'history.csv', index=False)
    pd.DataFrame({'trajectory_id': ids, 'true_soh': labels,
                  'predicted_soh': best_values}).to_csv(args.output / 'predictions.csv', index=False)
    summary = {'diagnostic': 'training-set memorization; not generalization',
               'passed': best_mae < 1.0, 'best_step': best_step,
               'mae_pp': best_mae, 'rmse_pp': best_rmse,
               'profile': 'square', 'initial_soc': 0.5, 'trajectory_ids': ids,
               'feature_columns': ['current_a', 'voltage_v'],
               'feature_mean': mean.tolist(), 'feature_std': std.tolist(),
               'device': str(device), 'source': str(source.resolve()),
               'config': {**vars(args), 'output': str(args.output)}}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Best MAE: {best_mae:.4f} pp at step {best_step}. Saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()

"""Train on real measurements with capacity-derived SOH labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from machine_learning.evaluate_real_soh import discharge_capacity, read_measurements, resample_segments
from machine_learning.models.lstm import LSTM
from machine_learning.models.mlp import MLP


def static_inputs(normalized_windows):
    """One endpoint vector per LSTM window; preserves exact prediction samples."""
    return normalized_windows[:, -1, :].copy()


def soh_metrics(targets, predictions):
    targets, predictions = np.asarray(targets, dtype=float), np.asarray(predictions, dtype=float)
    if targets.shape != predictions.shape or targets.size == 0:
        raise ValueError('Metrics require equally shaped nonempty predictions and labels')
    errors = predictions-targets
    if not np.isfinite(errors).all():
        raise ValueError('Nonfinite metrics inputs')
    variance = np.sum((targets-targets.mean())**2)
    return {'rmse_pp': float(100*np.sqrt(np.mean(errors**2))),
            'mae_pp': float(100*np.mean(abs(errors))),
            'r2': float(1-np.sum(errors**2)/variance) if variance > 1e-15 else None}


def predict_soh(model, loader, device, is_mlp=False):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for x, _ in loader:
            output = model(x.to(device))
            predictions.extend((output if is_mlp else output[1]).cpu().flatten().tolist())
    return np.asarray(predictions)


def mlp_candidates(args):
    base = {'hidden_layers': args.hidden_layers, 'hidden_dim': args.hidden_dim,
            'dropout': args.dropout, 'activation': args.activation,
            'learning_rate': args.learning_rate}
    if not args.mlp_search:
        return [base]
    # Six predeclared candidates, not a Cartesian product or test-guided sweep.
    return [dict(base, hidden_layers=1, hidden_dim=32, dropout=0., learning_rate=.001),
            dict(base, hidden_layers=2, hidden_dim=64, dropout=0., learning_rate=.001),
            dict(base, hidden_layers=3, hidden_dim=128, dropout=0., learning_rate=.001),
            dict(base, hidden_layers=2, hidden_dim=64, dropout=.1, learning_rate=.001),
            dict(base, hidden_layers=2, hidden_dim=64, dropout=0., learning_rate=.0003),
            dict(base, hidden_layers=2, hidden_dim=64, dropout=0., learning_rate=.003)]


def run_mlp(args, raw_train, raw_validation, train_labels, val_labels, mean, std, metadata, device):
    def loader(windows, labels, shuffle=False, static=True):
        normalized = (windows-mean)/std
        features = static_inputs(normalized) if static else normalized
        dataset = TensorDataset(torch.tensor(features, dtype=torch.float32),
                                torch.tensor(labels, dtype=torch.float32).unsqueeze(1))
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    train_eval = loader(raw_train, train_labels)
    validation = loader(raw_validation, val_labels)
    best_score, best_state, best_config, best_epoch = float('inf'), None, None, None
    trials, history = [], []
    for trial, config in enumerate(mlp_candidates(args)):
        torch.manual_seed(args.seed)
        train = loader(raw_train, train_labels, shuffle=True)
        model_config = {k: v for k, v in config.items() if k != 'learning_rate'}
        model = MLP(**model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        trial_score, trial_epoch = float('inf'), None
        for epoch in range(1, args.epochs+1):
            model.train()
            total_loss = 0.
            for x, y in train:
                optimizer.zero_grad()
                loss = (model(x.to(device))-y.to(device)).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite MLP loss')
                loss.backward()
                optimizer.step()
                total_loss += loss.item()*len(x)
            validation_metrics = soh_metrics(val_labels, predict_soh(model, validation, device, True))
            score = validation_metrics['rmse_pp']
            history.append({'trial': trial, 'epoch': epoch, 'training_mse': total_loss/len(raw_train),
                            **{f'validation_{k}': v for k, v in validation_metrics.items()}})
            if score < trial_score:
                trial_score, trial_epoch = score, epoch
            if score < best_score:
                best_score, best_config, best_epoch = score, dict(config), epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
                print(f'MLP trial {trial+1} epoch {epoch}/{args.epochs}: '
                      f'MSE {total_loss/len(raw_train):.6f}; val RMSE {score:.3f} pp', flush=True)
        trials.append({'trial': trial, **config, 'best_epoch': trial_epoch, 'validation_rmse_pp': trial_score})
        print(f'Trial {trial+1}: {config}; best val RMSE {trial_score:.3f} pp at epoch {trial_epoch}', flush=True)
    model_config = {k: v for k, v in best_config.items() if k != 'learning_rate'}
    best_model = MLP(**model_config).to(device)
    best_model.load_state_dict(best_state)
    torch.save({'model_type': 'mlp', 'model_state_dict': best_state, 'model_config': model_config,
                'training_config': best_config, 'epoch': best_epoch,
                'feature_columns': ['current_a', 'voltage_v'], 'feature_mean': mean.tolist(),
                'feature_std': std.tolist(), 'representation': 'last time step of normalized window',
                'sequence_length': 1800, 'training_metadata': metadata}, args.output/'best.pt')
    pd.DataFrame(trials).to_csv(args.output/'search.csv', index=False)
    pd.DataFrame(history).to_csv(args.output/'history.csv', index=False)
    val_predictions = predict_soh(best_model, validation, device, True)
    pd.DataFrame({'true_soh': val_labels, 'predicted_soh': val_predictions}).to_csv(
        args.output/'best_validation_predictions.csv', index=False)
    mean_target = float(np.mean(train_labels))
    report = {'best_config': best_config, 'best_epoch': best_epoch,
              'selection': 'validation RMSE; no early stopping in existing pipeline',
              'representation': 'current and voltage at last time step of each existing 1800-s window',
              'mlp': {'training': soh_metrics(train_labels, predict_soh(best_model, train_eval, device, True)),
                      'validation': soh_metrics(val_labels, val_predictions), 'test': None},
              'mean': {'training_mean_soh': mean_target,
                       'validation': soh_metrics(val_labels, np.full(len(val_labels), mean_target)), 'test': None},
              'lstm': None,
              'test_status': 'No test split exists in the current real-data pipeline; not repurposing validation.',
              'limitations': ['One cell, two checkpoint labels; exploratory same-cell comparison.',
                              'MLP observes endpoint only; LSTM observes current/voltage history.',
                              'Validation is reused for tuning; not an unbiased final test.']}
    comparison = None
    if args.compare_lstm_checkpoint:
        comparison = torch.load(args.compare_lstm_checkpoint, map_location=device, weights_only=True)
        prior = comparison['training_metadata']
        if prior['label_files'] != metadata['label_files'] or prior['split'] != metadata['split'] or prior.get('fresh_mode_files') != metadata.get('fresh_mode_files'):
            raise ValueError('LSTM checkpoint labels/source paths/split do not match this run')
        if any(prior[key] != metadata[key] for key in ['train_windows', 'validation_windows']):
            raise ValueError('LSTM sample counts differ')
        if not np.allclose(comparison['feature_mean'], mean, atol=1e-12, rtol=0) or not np.allclose(
            comparison['feature_std'], std, atol=1e-12, rtol=0
        ):
            raise ValueError('LSTM training normalization differs')
        lstm = LSTM(**comparison['model_config']).to(device)
        lstm.load_state_dict(comparison['model_state_dict'])
        val_lstm = predict_soh(lstm, loader(raw_validation, val_labels, static=False), device)
        report['lstm'] = {'checkpoint': str(args.compare_lstm_checkpoint.resolve()),
                          'epoch': comparison['epoch'], 'selection': 'existing validation-MAE-selected checkpoint',
                          'validation': soh_metrics(val_labels, val_lstm), 'test': None}
        report['limitations'].append('Historical LSTM selected by MAE; new MLP selected by RMSE.')
        pd.DataFrame({'true_soh': val_labels, 'mlp_prediction': val_predictions,
                      'lstm_prediction': val_lstm, 'mean_prediction': mean_target}).to_csv(
            args.output/'validation_comparison.csv', index=False)
    # Optional explicit test files are consumed ONLY after validation selection.
    if args.test_input:
        test_parts, test_labels = [], []
        for path, cap in zip(args.test_input, args.test_capacity_csv):
            target, _, _ = comparable_target(cap, args.bol_capacity_csv)
            part = load_windows(path)
            test_parts.append(part)
            test_labels.extend([target]*len(part))
        raw_test = np.concatenate(test_parts)
        test_predictions = predict_soh(best_model, loader(raw_test, test_labels), device, True)
        report['mlp']['test'] = soh_metrics(test_labels, test_predictions)
        report['mean']['test'] = soh_metrics(test_labels, np.full(len(test_labels), mean_target))
        columns = {'true_soh': test_labels, 'mlp_prediction': test_predictions,
                   'mean_prediction': np.full(len(test_labels), mean_target)}
        if comparison:
            test_lstm = predict_soh(lstm, loader(raw_test, test_labels, static=False), device)
            columns['lstm_prediction'] = test_lstm
            report['lstm']['test'] = soh_metrics(test_labels, test_lstm)
        pd.DataFrame(columns).to_csv(args.output/'test_predictions.csv', index=False)
        report['test_status'] = 'Explicit separate test files; same test windows for every model.'
    print('Best MLP configuration:', best_config, 'epoch', best_epoch, flush=True)
    print('Split / model        RMSE(pp)  MAE(pp)   R2', flush=True)
    for split in ['validation', 'test']:
        for name in ['mlp', 'lstm', 'mean']:
            result = report[name][split] if report.get(name) else None
            if result is None:
                print(f'{split} / {name}: unavailable', flush=True)
            else:
                r2 = 'undefined' if result['r2'] is None else f"{result['r2']:.4f}"
                print(f"{split} / {name}: {result['rmse_pp']:.4f}  {result['mae_pp']:.4f}  {r2}", flush=True)
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')


def load_windows(path, length=1800, fraction=None):
    windows = []
    for times, features in resample_segments(read_measurements(path), 1.0, 65.0):
        if fraction is not None:
            left, right = fraction
            features = features[int(len(features)*left):int(len(features)*right)]
            times = times[int(len(times)*left):int(len(times)*right)]
        for start in range(0, len(times) - length + 1, length):
            windows.append(features[start:start + length])
    if not windows:
        raise ValueError(f'No complete windows in {path}')
    return np.stack(windows)


def comparable_target(capacity_path, bol_path):
    capacity, _ = discharge_capacity(capacity_path)
    bol_capacity, _ = discharge_capacity(bol_path)
    rates = []
    cutoffs = []
    for path in [capacity_path, bol_path]:
        data = read_measurements(path)
        active = data['Current (A)'] < -0.05
        rates.append(float(-data.loc[active, 'Current (A)'].median()))
        cutoffs.append(float(data.loc[active, 'Voltage (V)'].min()))
    if not np.isclose(rates[0], rates[1], rtol=0.05):
        raise ValueError(f'Capacity discharge rates differ: {rates[0]:.3f} vs {rates[1]:.3f} A. '
                         'Supply a beginning-of-life capacity file at the same test rate.')
    if not np.isclose(cutoffs[0], cutoffs[1], atol=0.02):
        raise ValueError('Capacity voltage cutoffs differ; supply comparable capacity tests')
    return capacity / bol_capacity, capacity, bol_capacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path.home() / 'Downloads')
    parser.add_argument('--input', type=Path, nargs='+', help='One incremental CSV per aging checkpoint')
    parser.add_argument('--capacity-csv', type=Path, nargs='+', help='Matching capacity CSVs, in input order')
    parser.add_argument('--bol-capacity-csv', type=Path, help='Same-cell fresh capacity test at matching rate')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', choices=['lstm', 'mlp'], default='lstm')
    parser.add_argument('--mlp-search', action='store_true')
    parser.add_argument('--hidden-layers', type=int, default=2)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--activation', choices=['relu', 'tanh', 'gelu'], default='relu')
    parser.add_argument('--compare-lstm-checkpoint', type=Path)
    parser.add_argument('--test-input', type=Path, nargs='+')
    parser.add_argument('--test-capacity-csv', type=Path, nargs='+')
    args = parser.parse_args()
    if args.output is None:
        args.output = Path('outputs/real_mlp_training' if args.model == 'mlp' else 'outputs/real_soh_training')
    if args.compare_lstm_checkpoint and args.compare_lstm_checkpoint.resolve() == (args.output/'best.pt').resolve():
        parser.error('Choose a separate MLP output directory to preserve the comparison LSTM checkpoint')
    if any([args.input, args.capacity_csv, args.bol_capacity_csv]) and not all(
        [args.input, args.capacity_csv, args.bol_capacity_csv]
    ):
        parser.error('--input, --capacity-csv, and --bol-capacity-csv must be supplied together')
    if args.input and len(args.input) != len(args.capacity_csv):
        parser.error('Supply one capacity CSV for each input CSV')
    if args.hidden_layers < 1 or args.hidden_dim < 1 or not 0 <= args.dropout < 1:
        parser.error('Invalid MLP layer count, hidden dimension, or dropout')
    if bool(args.test_input) != bool(args.test_capacity_csv) or (args.test_input and len(args.test_input) != len(args.test_capacity_csv)):
        parser.error('Supply matching --test-input and --test-capacity-csv lists')
    if args.test_input and (args.model != 'mlp' or not args.input):
        parser.error('Explicit test evaluation requires --model mlp and labeled input files')
    if args.test_input and set(p.resolve() for p in args.test_input) & set(p.resolve() for p in args.input):
        parser.error('Test input files must not appear in training/validation inputs')
    if args.model != 'mlp' and (args.mlp_search or args.compare_lstm_checkpoint):
        parser.error('MLP search/comparison flags require --model mlp')
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('epochs, batch-size, and learning-rate must be positive')
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_path = args.data_dir / 'SAMSUNG_Cell_9_incremental_1C_Channel_3_Wb_1.CSV'
    validation_path = args.data_dir / 'SAMSUNG_Cell_9_incremental_01C_Channel_3_Wb_1.CSV'
    capacity_path = args.data_dir / 'SAMSUNG_Cell_9_capacity_02C_Channel_3_Wb_1.CSV'
    # These supplied files are from the 0-cycle folder. This is the BOL reference
    # itself, not a label inferred from current/voltage or a synthetic model.
    label_metadata = []
    if args.input:
        train_parts, val_parts, train_labels, val_labels = [], [], [], []
        for path, label_path in zip(args.input, args.capacity_csv):
            target, capacity, bol_capacity = comparable_target(label_path, args.bol_capacity_csv)
            # Split raw time BEFORE windowing, so no samples cross the boundary.
            train_part = load_windows(path, fraction=(0, .8))
            val_part = load_windows(path, fraction=(.8, 1))
            train_parts.append(train_part)
            val_parts.append(val_part)
            train_labels.extend([target] * len(train_part))
            val_labels.extend([target] * len(val_part))
            label_metadata.append({'input': str(path.resolve()), 'capacity_file': str(label_path.resolve()),
                                   'capacity_ah': capacity, 'bol_capacity_ah': bol_capacity, 'soh': target})
        raw_train, raw_validation = np.concatenate(train_parts), np.concatenate(val_parts)
        split = 'first 80%/last 20% of each gap-separated trace, before windowing; same-cell diagnostic'
    else:
        capacity, _ = discharge_capacity(capacity_path)
        target = 1.0
        raw_train, raw_validation = load_windows(train_path), load_windows(validation_path)
        train_labels, val_labels = [target]*len(raw_train), [target]*len(raw_validation)
        split = 'different protocols from SAME fresh cell; no test set'
    mean = raw_train.mean(axis=(0, 1))
    std = raw_train.std(axis=(0, 1))
    if not np.isfinite([mean, std]).all() or (std <= 0).any():
        raise ValueError('Invalid training normalization')

    def make_loader(raw, labels, shuffle):
        x = torch.tensor((raw - mean) / std, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        return DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(raw_train, train_labels, True)
    train_eval_loader = make_loader(raw_train, train_labels, False)
    validation_loader = make_loader(raw_validation, val_labels, False)
    model_config = {'input_size': 2, 'hidden_size': 64, 'num_layers': 1}
    model = LSTM(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    def evaluate(loader):
        model.eval()
        predictions, targets = [], []
        with torch.inference_mode():
            for x, y in loader:
                predictions.extend(model(x.to(device))[1].cpu().flatten().tolist())
                targets.extend(y.flatten().tolist())
        errors = np.asarray(predictions) - np.asarray(targets)
        return float(100 * np.abs(errors).mean()), predictions

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        'diagnostic': 'real-data fitting; same-cell validation does not measure unseen-cell generalization',
        'label_files': label_metadata,
        'fresh_mode_files': None if args.input else [str(train_path.resolve()), str(validation_path.resolve()), str(capacity_path.resolve())],
        'unique_soh_labels': sorted(set(train_labels)),
        'label_basis': 'comparable capacity / fresh capacity' if args.input else 'fresh checkpoint SOH=1',
        'mean_baseline_validation_mae_pp': float(100*np.mean(abs(np.asarray(val_labels)-np.mean(train_labels)))),
        'split': split,
        'feature_mean': mean.tolist(), 'feature_std': std.tolist(),
        'resampling': '1 s; held current; interpolated voltage; split gaps >65 s',
        'window_length': 1800, 'stride': 1800, 'train_windows': len(raw_train),
        'validation_windows': len(raw_validation),
        'config': {key: str(value) if isinstance(value, Path) else
                   [str(p) for p in value] if isinstance(value, list) else value
                   for key, value in vars(args).items()},
    }
    (args.output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Training {args.model.upper()} FROM SCRATCH on REAL current/voltage data; device={device}', flush=True)
    print(f'Train: {len(raw_train)} windows; validation: {len(raw_validation)} windows', flush=True)
    print('SOH labels:', sorted(set(train_labels)), flush=True)
    print('Split:', split, flush=True)
    if len(set(train_labels)) == 1:
        print('SINGLE HEALTH LABEL: constant prediction is sufficient. Convergence diagnostic only.', flush=True)
    if args.model == 'mlp':
        run_mlp(args, raw_train, raw_validation, train_labels, val_labels, mean, std, metadata, device)
        return
    best = float('inf')
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = (model(x)[1] - y).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        train_mae, _ = evaluate(train_eval_loader)
        val_mae, predictions = evaluate(validation_loader)
        history.append({'epoch': epoch, 'training_mse': total_loss / len(raw_train),
                        'training_mae_pp': train_mae, 'validation_mae_pp': val_mae})
        pd.DataFrame(history).to_csv(args.output / 'history.csv', index=False)
        if val_mae < best:
            best = val_mae
            torch.save({'model_state_dict': model.state_dict(), 'model_config': model_config,
                        'feature_columns': ['current_a', 'voltage_v'],
                        'feature_mean': mean.tolist(), 'feature_std': std.tolist(),
                        'sequence_length': 1800, 'sample_interval_s': 1.0,
                        'current_convention': 'positive_charge', 'epoch': epoch,
                        'validation_soh_mae_pp': best, 'training_metadata': metadata},
                       args.output / 'best.pt')
            pd.DataFrame({'true_soh': val_labels, 'predicted_soh': predictions}).to_csv(
                args.output / 'best_validation_predictions.csv', index=False)
        print(f'Epoch {epoch}/{args.epochs} | SOH loss: {total_loss / len(raw_train):.6f} | '
              f'Train MAE: {train_mae:.3f} pp | Validation MAE: {val_mae:.3f} pp', flush=True)
    print(f'Best validation MAE: {best:.3f} pp. Saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()

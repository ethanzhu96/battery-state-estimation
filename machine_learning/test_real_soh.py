"""Checks for the real-data adapter's time and label handling."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import json

import numpy as np
import pandas as pd
import torch

from machine_learning.evaluate_real_soh import (
    CURRENT, TIME, VOLTAGE, discharge_capacity, read_measurements, resample_segments,
)
from machine_learning.models.mlp import MLP
from machine_learning.train_real_soh import static_inputs, soh_metrics, run_mlp


class RealSOHTests(unittest.TestCase):
    def test_test_data_does_not_select_mlp_or_fit_scaling(self):
        train = np.array([[[0., 3.], [1., 4.]], [[2., 5.], [3., 6.]]])
        validation = train + .1
        mean, std = train.mean(axis=(0, 1)), train.std(axis=(0, 1))
        with tempfile.TemporaryDirectory() as directory:
            saved = []
            for index, shift in enumerate([0., 100.]):
                output = Path(directory)/str(index)
                output.mkdir()
                args = SimpleNamespace(batch_size=2, seed=42, epochs=2, output=output,
                                       mlp_search=False, hidden_layers=1, hidden_dim=8, dropout=0.,
                                       activation='relu', learning_rate=.001, compare_lstm_checkpoint=None,
                                       test_input=[Path('separate.csv')], test_capacity_csv=[Path('capacity.csv')],
                                       bol_capacity_csv=Path('fresh.csv'))
                with patch('machine_learning.train_real_soh.load_windows', return_value=train+shift), \
                     patch('machine_learning.train_real_soh.comparable_target', return_value=(.9, 1.8, 2.)), \
                     redirect_stdout(io.StringIO()):
                    run_mlp(args, train, validation, [.8, 1.], [.8, 1.], mean, std, {}, torch.device('cpu'))
                checkpoint = torch.load(output/'best.pt', weights_only=True)
                np.testing.assert_equal(checkpoint['feature_mean'], mean)
                np.testing.assert_equal(checkpoint['feature_std'], std)
                saved.append((checkpoint, json.loads((output/'metrics.json').read_text())))
            self.assertEqual(saved[0][1]['mlp']['validation'], saved[1][1]['mlp']['validation'])
            self.assertIsNotNone(saved[0][1]['mlp']['test'])
            self.assertNotEqual(saved[0][1]['mlp']['test'], saved[1][1]['mlp']['test'])
            for key in saved[0][0]['model_state_dict']:
                torch.testing.assert_close(saved[0][0]['model_state_dict'][key], saved[1][0]['model_state_dict'][key])

    def test_mlp_observes_endpoint_only(self):
        windows = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
        endpoints = static_inputs(windows)
        windows[:, 0, :] += 100
        np.testing.assert_equal(endpoints, static_inputs(windows))
        for activation in ['relu', 'tanh', 'gelu']:
            model = MLP(hidden_layers=3, hidden_dim=32, dropout=.1, activation=activation)
            output = model(torch.tensor(endpoints, dtype=torch.float32))
            self.assertEqual(output.shape, (2, 1))
            output.sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            with self.assertRaisesRegex(ValueError, 'static'):
                model(torch.tensor(windows, dtype=torch.float32))

    def test_metrics_and_mean_baseline(self):
        target = [.7, .8, .9, 1.]
        result = soh_metrics(target, [.85]*4)
        self.assertAlmostEqual(result['mae_pp'], 10.)
        self.assertAlmostEqual(result['rmse_pp'], 100*np.sqrt(.0125))
        self.assertAlmostEqual(result['r2'], 0.)
        self.assertEqual(soh_metrics(target, target)['r2'], 1.)
        self.assertIsNone(soh_metrics([1., 1.], [1., 1.])['r2'])

    def test_current_steps_and_gap_boundaries(self):
        data = pd.DataFrame({TIME: [0., 2., 4., 100., 102.],
                             CURRENT: [0., -2., 0., 1., 1.],
                             VOLTAGE: [4., 3.8, 4., 4., 4.2]})
        segments = list(resample_segments(data, 1., 5.))
        self.assertEqual(len(segments), 2)
        np.testing.assert_equal(segments[0][0], [0, 1, 2, 3])
        np.testing.assert_equal(segments[0][1][:, 0], [0, 0, -2, -2])
        np.testing.assert_allclose(segments[0][1][:, 1], [4, 3.9, 3.8, 3.9])
        self.assertEqual(segments[1][0][0], 100)

    def test_capacity_is_increment_not_absolute_counter(self):
        data = pd.DataFrame({TIME: [0, 1, 2, 3], CURRENT: [0, -1, -1, 0],
                             VOLTAGE: [4.2, 4., 2.75, 3.],
                             'Discharge Capacity (Ah)': [5., 5.5, 7., 7.]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'capacity.csv'
            data.to_csv(path, index=False)
            self.assertEqual(discharge_capacity(path)[0], 2.)
            data.loc[2, 'Discharge Capacity (Ah)'] = 0
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'resets'):
                discharge_capacity(path)

    def test_reject_time_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reset.csv'
            pd.DataFrame({TIME: [0, 2, 1], CURRENT: [0, 0, 0],
                          VOLTAGE: [4, 4, 4]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'strictly increase'):
                read_measurements(path)


if __name__ == '__main__':
    unittest.main()

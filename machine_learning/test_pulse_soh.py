import unittest

import numpy as np
import pandas as pd
import torch

from machine_learning.compare_pulse_soh import PulseLSTM, extract_pulses, match_pulses, relative_voltage, ridge_predict
from machine_learning.models.lstm import LSTM
from machine_learning.evaluate_real_soh import CURRENT, TIME, VOLTAGE


class PulseTests(unittest.TestCase):
    def test_resistance_and_window_alignment(self):
        t = np.arange(0., 181., .1)
        current = np.where((t >= 10) & (t < 20), -2., 0.)
        frame = pd.DataFrame({TIME: t, CURRENT: current, VOLTAGE: 4.+.05*current})
        events, windows = extract_pulses(frame)
        self.assertEqual(len(events), 1)
        self.assertEqual(windows.shape, (1, 156, 2))
        self.assertAlmostEqual(events.r_on_ohm.iloc[0], .05)
        self.assertAlmostEqual(events.r_off_ohm.iloc[0], .05)
        self.assertAlmostEqual(events.recovery_120_v.iloc[0], 0.)
        np.testing.assert_equal(windows[0, :5, 0], 0.)
        self.assertEqual(windows[0, 5, 0], -2.)

    def test_matching_does_not_reuse_events_or_exceed_tolerance(self):
        a = pd.DataFrame({'pre_voltage_v': [4., 3.9, 3.8],
                          'current_a': [-2.]*3, 'duration_s': [10.]*3})
        b = pd.DataFrame({'pre_voltage_v': [3.99, 3.89, 3.5],
                          'current_a': [-2.]*3, 'duration_s': [10.]*3})
        self.assertEqual(match_pulses(a, b, .04), [(0, 0), (1, 1)])

    def test_ridge_handles_constant_feature(self):
        predictions = ridge_predict(np.ones((4, 1)), np.ones(4)*.95, np.ones((2, 1)))
        np.testing.assert_allclose(predictions, [.95, .95])

    def test_relative_voltage_removes_constant_offset_without_mutation(self):
        sequence = np.array([[[0., 4.], [-2., 3.9], [0., 3.95]]])
        original = sequence.copy()
        relative = relative_voltage(sequence, [4.])
        shifted = sequence.copy()
        shifted[:, :, 1] += .3
        np.testing.assert_allclose(relative, relative_voltage(shifted, [4.3]), atol=1e-12)
        np.testing.assert_equal(sequence, original)
        np.testing.assert_equal(relative[:, :, 0], sequence[:, :, 0])

    def test_original_model_is_preserved_and_feature_head_gets_gradients(self):
        x = torch.randn(3, 8, 2)
        torch.manual_seed(42)
        original = LSTM()
        torch.manual_seed(42)
        baseline = PulseLSTM()
        torch.testing.assert_close(original(x)[1], baseline(x)[1])
        for pooling in ['mean', 'last']:
            model = PulseLSTM(pooling=pooling, feature_count=4)
            predictions = model(x, torch.randn(3, 4))[1]
            self.assertEqual(predictions.shape, (3, 1))
            predictions.square().sum().backward()
            self.assertTrue(torch.isfinite(model.lstm.weight_ih_l0.grad).all())
            self.assertGreater(model.soh_head.weight.grad[:, -4:].abs().sum().item(), 0)
            with self.assertRaisesRegex(ValueError, 'feature vector'):
                model(x)


if __name__ == '__main__':
    unittest.main()

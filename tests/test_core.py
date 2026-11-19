"""Small CPU checks; synthetic data do not validate paper accuracy."""
import unittest
import numpy as np
import torch

from refit.core import HybridVote59, Learner59, Schedule, DAUCBv2, macro_f1
from refit.model import MLPModel


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_query_only_recovery_and_distillation(self):
        np.random.seed(3)
        torch.manual_seed(3)
        x = np.random.randn(256, 6).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.int64)
        learner = Learner59(6, [[16, 8], [8, 4]], MLPModel, Schedule([1]),
                            device='cpu', train_epochs=1, teacher_epochs=1,
                            warmup_windows=1, exemplar_size=16, batch_size=64, ucb_version='v2')
        learner.initialize(x, y)
        self.assertEqual(learner.ucb_version, 'v2')
        self.assertEqual(learner.ucb.n.sum(), 0)
        learner.label_budget = 8
        window = x + .5
        labels = (window[:, 0] > 0).astype(np.int64)
        queries = []

        def oracle(indices):
            queries.append(indices.copy())
            return labels[indices]

        frozen = learner.predict(window)
        pre, post, info = learner.recover_window(window, oracle)
        np.testing.assert_array_equal(pre, frozen)
        self.assertEqual(len(queries), 1)
        self.assertEqual(len(np.unique(queries[0])), 8)
        self.assertEqual(len(learner.pool_y), len(y) + 8)
        self.assertEqual(info['reward']['timing'], 'pre_update_deployed_arch')
        self.assertEqual(info['reward']['arch'], 0)
        self.assertAlmostEqual(info['reward']['value'], macro_f1(labels[queries[0]], frozen[queries[0]]))
        self.assertEqual(learner.ucb.n.sum(), 1)
        self.assertTrue(info['distilled'])
        self.assertTrue(info['teacher_updated'])
        self.assertEqual(post.shape, labels.shape)
        rewards = learner.ucb.n.copy()

        def forbidden_oracle(indices):
            self.fail('Untriggered window must not query labels')

        pre, post, info = learner.recover_window(window, forbidden_oracle)
        np.testing.assert_array_equal(pre, post)
        np.testing.assert_array_equal(learner.ucb.n, rewards)
        self.assertEqual(len(info['queried']), 0)

    def test_detector_and_ucb(self):
        np.random.seed(1)
        torch.manual_seed(1)
        x = np.random.randn(64, 6).astype(np.float32)
        detector = HybridVote59(6, device='cpu', train_epochs=1)
        detector.fit(x)
        for shift in [0., 1., 2., 3., 4.]:
            trigger, info = detector.detect(x + shift)
            self.assertEqual(trigger, info['votes'] >= 2)
            self.assertGreaterEqual(info['severity'], 0)
            self.assertLessEqual(info['severity'], 1)
        self.assertIsNotNone(info['quantile']['threshold'])
        bandit = DAUCBv2(2)
        self.assertEqual(bandit.select(False), 0)
        bandit.update(0, .5)
        self.assertEqual(bandit.select(True), 1)
        bandit.update(1, .25)
        bandit.select(0)
        low = np.asarray(bandit.last_values)
        bandit.select(1)
        self.assertTrue(np.all(np.asarray(bandit.last_values) > low))
        self.assertEqual(macro_f1(np.array([0, 0]), np.array([0, 0])), 1.)


if __name__ == '__main__':
    unittest.main()

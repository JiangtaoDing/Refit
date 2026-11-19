import unittest
import numpy as np
import torch
from refit.core import DAUCB59, DAUCBv2, Learner59, macro_f1, binary_f1
from refit.model import MLPModel


class Trigger:
    def __init__(self, severity=.5): self.severity = severity
    def fit(self, X): pass
    def detect(self, X): return True, dict(votes=3, severity=self.severity)


class MacroF1Test(unittest.TestCase):
    def test_perfect_single_class_is_one(self):
        z = np.zeros(10, int)
        self.assertEqual(macro_f1(z, z), 1.0); self.assertEqual(macro_f1(z + 1, z + 1), 1.0)
        self.assertEqual(binary_f1(z, z), 0.0)  # the v1 defect on all-benign queries

    def test_class_symmetric(self):
        y = np.array([0, 0, 0, 1, 1]); p = np.array([0, 0, 1, 1, 1])
        self.assertAlmostEqual(macro_f1(y, p), macro_f1(1 - y, 1 - p))


class SelectorTest(unittest.TestCase):
    def play(self, good, T=14):
        u = DAUCBv2(4); arm = 0; seq = []
        for _ in range(T):
            u.update(arm, 1. if arm == good else 0.); arm = u.select(.5); seq.append(arm)
        return seq

    def test_selection_depends_on_rewards(self):
        a, b = self.play(1), self.play(3)
        self.assertNotEqual(a, b)
        self.assertGreater(a[4:].count(1), a[4:].count(3)); self.assertGreater(b[4:].count(3), b[4:].count(1))

    def test_severity_scales_exploration_not_constant(self):
        # Arm 0: high mean, many pulls. Arm 1: lower mean, one pull. Low severity exploits,
        # high severity explores; an additive constant (v1) could never flip this.
        u = DAUCBv2(2, beta=2.0); u.n[:] = [30, 10]; u.r_hat[:] = [.95, .30]
        self.assertEqual(u.select(0.0), 0); self.assertEqual(u.select(1.0), 1)
        v1 = DAUCB59(2, beta=2.0); v1.n[:] = [30, 10]; v1.r_hat[:] = [.95, .30]; v1.t = 40
        self.assertEqual(v1.select(False), v1.select(True))


class LearnerV2Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0); np.random.seed(0)
        self.X = np.random.randn(400, 4).astype('float32'); self.y = (self.X[:, 0] > 0).astype('int64')

    def make(self, version):
        L = Learner59(4, [[8, 4], [4, 8]], MLPModel, Trigger(), device='cpu', train_epochs=1, teacher_epochs=1,
                      batch_size=64, warmup_windows=100, exemplar_size=50, ucb_version=version)
        L.initialize(self.X, self.y); L.label_budget = 20
        return L

    def test_delayed_reward_uses_deployed_pre_predictions(self):
        L = self.make('v2')
        self.assertEqual(L.ucb.n.sum(), 0)  # no reward from initialization labels
        pre, post, info = L.recover_window(self.X, lambda idx: self.y[idx])
        q = info['queried']
        self.assertEqual(info['reward']['arch'], 0)
        self.assertAlmostEqual(info['reward']['value'], macro_f1(self.y[q], pre[q]))
        self.assertEqual(L.ucb.n.tolist(), [1., 0.])
        self.assertEqual(info['next_arch'], [4, 8])  # the unrewarded arm is tried next

    def test_v1_keeps_original_reward(self):
        L = self.make('v1')
        self.assertEqual(L.ucb.n.sum(), 1)  # 59 rewards the initialization fit
        pre, post, info = L.recover_window(self.X, lambda idx: self.y[idx])
        q = info['queried']
        self.assertAlmostEqual(info['reward']['value'], binary_f1(self.y[q], post[q]))


class R16VariantTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0); np.random.seed(0)
        self.X = np.random.randn(400, 4).astype('float32'); self.y = (self.X[:, 0] > 0).astype('int64')

    def make(self, archs=([8, 4], [4, 8]), **kw):
        L = Learner59(4, [list(a) for a in archs], MLPModel, Trigger(), device='cpu', train_epochs=1, teacher_epochs=1,
                      batch_size=64, warmup_windows=1, exemplar_size=50, ucb_version='v2', **kw)
        L.initialize(self.X, self.y); L.label_budget = 20
        return L

    def test_no_replay_trains_on_query_set_only(self):
        L = self.make(replay=False)
        pre, post, info = L.recover_window(self.X, lambda idx: self.y[idx])
        self.assertEqual(info['train_size'], len(info['queried']))
        full = self.make()
        _, _, info2 = full.recover_window(self.X, lambda idx: self.y[idx])
        self.assertEqual(info2['train_size'], 400 + 20 + 50)  # W0 pool + Q_t + exemplar

    def test_no_distill_skips_teacher(self):
        L = self.make(distill=False); L.window_idx = 5
        _, _, info = L.recover_window(self.X, lambda idx: self.y[idx])
        self.assertFalse(info['distilled']); self.assertFalse(info['teacher_updated'])
        D = self.make(); D.window_idx = 5
        _, _, info = D.recover_window(self.X, lambda idx: self.y[idx])
        self.assertTrue(info['distilled'])

    def test_fixed_arch_never_switches(self):
        L = self.make(archs=([8, 4],))
        for _ in range(3):
            _, _, info = L.recover_window(self.X, lambda idx: self.y[idx])
            self.assertEqual(info['next_arch'], [8, 4])


if __name__ == '__main__':
    unittest.main()

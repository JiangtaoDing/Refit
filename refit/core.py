"""ReFIT strict-59: the original 59 pipeline (对比方案/RECDA/59_complete_pipeline_20260330)
with every use of non-queried labels from the current window removed.

Kept from 59 (unchanged logic):
  * Hybrid_Vote detector: fixed MAD vs W0, sliding MAD over previous windows (>=3),
    quantile over the last 20 window medians (>=5 previous), 2/3 vote.
  * Mixed sampling: cluster-stratified pre-selection (30%/20%/5% by cluster size) as the
    anomaly-like indicator, 0.6*indicator + 0.4*entropy, top-K.
  * Training set = cumulative labeled pool (all of W0 + every queried sample) + exemplar.
  * Teacher incremental update (lr 1e-4) + student distillation after warm-up window 10.
  * DA-UCB of 59: r_hat + sqrt(2*alpha*log t / n) + beta*drift; no soft reset (59 never calls it).
  * Recovered student re-scores the current window (post-update evaluation).

Strict changes (label leakage removed):
  * UCB reward after a recovery = F1 of the recovered student on the queried subset Q_t only
    (manuscript Eq. R_t). 59 used full-window F1 and also rewarded untriggered windows; here
    untriggered windows give no reward (no labels exist for them).
  * Post-update metrics are reported both on the full window and on the unqueried remainder
    (within-window protocol); pre-update frozen predictions are also stored.
Hyper-parameters follow the manuscript table (p=0.9, rho=0.75, tau=3, K=ceil(1%|W_t|)).
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32))
        self.decoder = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, d))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class HybridVote59:
    """Verbatim decision logic of the 59 detector."""

    def __init__(self, d, device='cuda', train_epochs=20, batch_size=512, fixed_threshold=3.0,
                 sliding_window_size=5, quantile_alpha=0.9, quantile_history=20, vote_min=2):
        self.device, self.train_epochs, self.batch_size = device, train_epochs, batch_size
        self.fixed_threshold, self.sliding_window_size = fixed_threshold, sliding_window_size
        self.quantile_alpha, self.quantile_len = quantile_alpha, quantile_history
        self.vote_min = vote_min
        self.ae = _AE(d).to(device)
        self.baseline_errors = None
        self.sliding_history, self.quantile_history = [], []

    def fit(self, X):
        loader = DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.ae.parameters(), lr=1e-3)
        self.ae.train()
        for _ in range(self.train_epochs):
            for (xb,) in loader:
                xb = xb.to(self.device)
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.ae(xb)[0], xb)
                loss.backward()
                opt.step()
        self.baseline_errors = self._errors(X)

    def _errors(self, X):
        self.ae.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 5000):
                xb = torch.as_tensor(X[i:i + 5000], dtype=torch.float32, device=self.device)
                out.append(torch.mean((xb - self.ae(xb)[0]) ** 2, dim=1).cpu().numpy())
        return np.concatenate(out)

    def detect(self, X):
        m = float(np.median(self._errors(X)))
        d1, s1 = False, 0.0
        ref = float(np.median(self.baseline_errors))
        mad = float(np.median(np.abs(self.baseline_errors - ref)))
        if mad >= 1e-6:
            s1 = (m - ref) / mad
            d1 = s1 > self.fixed_threshold
        self.sliding_history.append(m)
        self.sliding_history = self.sliding_history[-self.sliding_window_size:]
        d2, s2, t2 = False, 0.0, None
        if len(self.sliding_history) >= 3:
            rec = np.array(self.sliding_history[:-1], dtype=np.float32)
            base = float(np.median(rec))
            mad = float(np.median(np.abs(rec - base)))
            sd = float(np.std(rec))
            t2 = 3.0 if sd < .01 else (2.5 if sd < .05 else 2.0)
            if mad >= 1e-6:
                s2 = (m - base) / mad
                d2 = s2 > t2
        self.quantile_history.append(m)
        self.quantile_history = self.quantile_history[-self.quantile_len:]
        d3, t3 = False, None
        if len(self.quantile_history) >= 5:
            t3 = float(np.quantile(self.quantile_history[:-1], self.quantile_alpha))
            d3 = m > t3
        votes = int(d1) + int(d2) + int(d3)
        # Label-free drift severity in [0,1] (used only by UCB v2): mean over the three
        # criteria of e/(1+e), e = relative exceedance of its own threshold (0 if undefined).
        ex = [max(s1 / self.fixed_threshold - 1, 0.0),
              max(s2 / t2 - 1, 0.0) if t2 else 0.0,
              max(m / t3 - 1, 0.0) if t3 else 0.0]
        severity = float(np.mean([e / (1 + e) for e in ex]))
        return votes >= self.vote_min, dict(median_error=m, votes=votes, severity=severity, fixed=dict(score=s1, drift=d1),
                                sliding=dict(score=s2, threshold=t2, drift=d2), quantile=dict(threshold=t3, drift=d3))


class Schedule:
    def __init__(self, windows):
        self.windows, self.i = set(windows), 0

    def fit(self, X):
        pass

    def detect(self, X):
        self.i += 1
        return self.i in self.windows, dict(schedule_window=self.i, votes=0, severity=0.0)


class DAUCB59:
    """59 DA-UCB (the drift bonus is identical for every arm)."""

    def __init__(self, k, alpha=2.0, beta=0.5):
        self.k, self.alpha, self.beta = k, alpha, beta
        self.n, self.r_hat, self.t = np.zeros(k), np.zeros(k), 0
        self.last_values = None

    def select(self, drift):
        self.t += 1
        v = np.full(self.k, np.inf)
        for i in range(self.k):
            if self.n[i] > 0:
                v[i] = self.r_hat[i] + np.sqrt(2 * self.alpha * np.log(self.t) / self.n[i]) + self.beta * float(drift)
        return int(np.argmax(v))

    def update(self, i, r):
        self.n[i] += 1
        self.r_hat[i] += (r - self.r_hat[i]) / self.n[i]


class DAUCBv2(DAUCB59):
    """Drift-aware UCB1: r_hat_k + (1 + beta * v_t) * sqrt(2 * alpha * ln(N + 1) / n_k).

    v_t in [0,1] is the label-free drift severity, so it scales exploration relative to
    the reward estimates instead of adding the same constant to every arm. N is the
    total number of rewards received; arms never rewarded are tried first.
    """

    def __init__(self, k, alpha=1.0, beta=0.5):
        super().__init__(k, alpha, beta)

    def select(self, severity):
        self.t += 1
        N = float(self.n.sum())
        sev = float(np.clip(severity, 0.0, 1.0))
        v = np.full(self.k, np.inf)
        for i in range(self.k):
            if self.n[i] > 0:
                v[i] = self.r_hat[i] + (1 + self.beta * sev) * np.sqrt(2 * self.alpha * np.log(N + 1) / self.n[i])
        self.last_values = [float(x) for x in v]
        return int(np.argmax(v))


def kd_loss(student, teacher, y, rho=0.75, tau=3.0):
    hard = nn.functional.cross_entropy(student, y)
    soft = nn.functional.kl_div(nn.functional.log_softmax(student / tau, 1), nn.functional.softmax(teacher / tau, 1),
                                reduction='batchmean') * tau ** 2
    return rho * soft + (1 - rho) * hard


def macro_f1(y, p):
    """Unweighted mean of per-class F1 over the classes present in y or p (class-symmetric;
    1.0 for a perfect prediction even when only one class occurs)."""
    s = 0.0; classes = np.union1d(np.unique(y), np.unique(p))
    for c in classes:
        tp = int(((y == c) & (p == c)).sum()); fp = int(((y != c) & (p == c)).sum()); fn = int(((y == c) & (p != c)).sum())
        s += 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return s / len(classes) if len(classes) else 0.0


def binary_f1(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


class Learner59:
    def __init__(self, d, archs, model_class, detector, device='cuda', train_epochs=20, teacher_epochs=10,
                 batch_size=512, warmup_windows=10, exemplar_size=2000, anomaly_weight=0.6, rho=0.75, tau=3.0,
                 selection='mixed', distill=True, ucb_version='v1', replay=True, gpu_data=False):
        self.d, self.archs, self.model_class, self.detector, self.device = d, archs, model_class, detector, device
        self.train_epochs, self.teacher_epochs, self.batch_size = train_epochs, teacher_epochs, batch_size
        self.warmup_windows, self.exemplar_size, self.anomaly_weight = warmup_windows, exemplar_size, anomaly_weight
        self.rho, self.tau, self.selection, self.distill = rho, tau, selection, distill
        # replay=False (R1.6 ablation): train on the current query set Q_t only, i.e. no
        # cumulative labeled pool and no exemplar memory.
        self.replay = replay
        # gpu_data=True (overhead timing only): keep the training set on the GPU and index it
        # with exactly the permutation DataLoader(shuffle=True) would draw from the global RNG,
        # so batches, RNG consumption and the training trajectory are unchanged.
        self.gpu_data = gpu_data
        if ucb_version not in ('v1', 'v2'):
            raise ValueError("ucb_version must be 'v1' or 'v2'")
        self.ucb_version = ucb_version
        # v1: the original 59 selector (reward of the recovered student on its own Q_t).
        # v2: delayed reward of the deployed architecture on Q_t from its pre-update
        #     predictions (Q_t unseen by that model), macro-F1, multiplicative drift term.
        self.ucb = DAUCB59(len(archs)) if ucb_version == 'v1' else DAUCBv2(len(archs))
        self.window_idx = 0
        self.label_budget = 0

    # ---------- model utilities ----------
    def _new(self, arch):
        return self.model_class(self.d, 2, hidden_dims=arch).to(self.device)

    def _loader(self, X, y):
        return DataLoader(TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)), batch_size=self.batch_size, shuffle=True)

    def _gpu_batches(self, Xg, yg):
        """Same RNG draws and batch order as iterating DataLoader(TensorDataset, shuffle=True)."""
        _shared = torch.empty((), dtype=torch.int64).random_()  # DataLoader iterator base seed
        seed = int(torch.empty((), dtype=torch.int64).random_().item())  # RandomSampler seed
        g = torch.Generator(); g.manual_seed(seed)
        perm = torch.randperm(len(Xg), generator=g).to(self.device)
        for i in range(0, len(Xg), self.batch_size):
            j = perm[i:i + self.batch_size]
            yield Xg[j], yg[j]

    def _train(self, model, X, y, epochs, lr, teacher=None):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        if teacher is not None:
            teacher.eval()
        if self.gpu_data:
            Xg = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
            yg = torch.as_tensor(np.asarray(y), dtype=torch.int64, device=self.device)
        for _ in range(epochs):
            for xb, yb in (self._gpu_batches(Xg, yg) if self.gpu_data else self._loader(X, y)):
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                out = model(xb)[0]
                if teacher is None:
                    loss = nn.functional.cross_entropy(out, yb)
                else:
                    with torch.no_grad():
                        t = teacher(xb)[0]
                    loss = kd_loss(out, t, yb, self.rho, self.tau)
                loss.backward()
                opt.step()
        return model

    def _probs(self, model, X):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 5000):
                out.append(torch.softmax(model(torch.as_tensor(X[i:i + 5000], dtype=torch.float32, device=self.device))[0], 1).cpu().numpy())
        return np.vstack(out)

    def predict(self, X, model=None):
        return self._probs(model or self.model, X).argmax(1)

    def _exemplar(self, X, y):
        if self.ex_X is None:
            cx, cy = X, y
        else:
            cx, cy = np.vstack([self.ex_X, X]), np.concatenate([self.ex_y, y])
        idx = np.random.choice(len(cx), min(self.exemplar_size, len(cx)), replace=False)
        self.ex_X, self.ex_y = cx[idx], cy[idx]

    # ---------- 59 mixed sampling ----------
    def _select(self, X):
        K = min(self.label_budget, len(X))
        if self.selection == 'random':
            return np.random.choice(len(X), K, replace=False)
        from sklearn.cluster import MiniBatchKMeans
        nc = min(20, len(X) // 100)
        if nc < 2:
            return np.random.choice(len(X), K, replace=False)
        labels = MiniBatchKMeans(n_clusters=nc, random_state=42, batch_size=1000).fit_predict(X)
        sel = []
        for c, size in zip(*np.unique(labels, return_counts=True)):
            ratio = .30 if size < len(X) * .05 else (.20 if size < len(X) * .10 else .05)
            m = max(1, int(size * ratio)); ids = np.flatnonzero(labels == c)
            sel.extend(np.random.choice(ids, m, replace=False) if len(ids) > m else ids)
        sel = np.asarray(sel)
        if len(sel) > K:
            sel = np.random.choice(sel, K, replace=False)
        elif len(sel) < K:
            rest = np.setdiff1d(np.arange(len(X)), sel)
            sel = np.concatenate([sel, np.random.choice(rest, min(K - len(sel), len(rest)), replace=False)])
        a = np.zeros(len(X), dtype=np.float32); a[sel] = 1.0
        p = self._probs(self.model, X); u = -np.sum(p * np.log(p + 1e-10), axis=1)
        a = (a - a.min()) / (a.max() - a.min() + 1e-10); u = (u - u.min()) / (u.max() - u.min() + 1e-10)
        return np.argsort(self.anomaly_weight * a + (1 - self.anomaly_weight) * u)[-K:]

    # ---------- protocol ----------
    def initialize(self, X, y):
        i = self.ucb.select(False) if self.ucb_version == 'v1' else 0
        self.arch = self.archs[i]
        self.model = self._train(self._new(self.arch), X, y, self.train_epochs, 1e-3)
        self.teacher = copy.deepcopy(self.model)
        self.ex_X = self.ex_y = None
        self._exemplar(X, y)
        self.detector.fit(X)
        self.pool_X, self.pool_y = X.copy(), y.copy()
        if self.ucb_version == 'v1':
            self.ucb.update(i, binary_f1(y, self.predict(X)))  # initialization labels only
        self.window_idx = 1

    def recover_window(self, X, oracle):
        """Returns (pre_pred, post_pred, info). Only oracle(idx) reveals labels of W_t."""
        pre = self.predict(X)
        deployed = list(self.arch)
        drift, dinfo = self.detector.detect(X)
        info = dict(trigger=bool(drift), votes=dinfo.get('votes', 0), deployed_arch=deployed, queried=np.empty(0, int),
                    teacher_updated=False, distilled=False, reward=None, train_size=0)
        if drift:
            idx = np.asarray(self._select(X), dtype=int)
            if len(idx) > self.label_budget or len(np.unique(idx)) != len(idx):
                raise ValueError('budget violation')
            lab = np.asarray(oracle(idx), dtype=np.int64)
            self.pool_X = np.vstack([self.pool_X, X[idx]]); self.pool_y = np.concatenate([self.pool_y, lab])
            if self.ucb_version == 'v2':
                # Delayed reward, computed before any training on Q_t.
                dep = self.archs.index(deployed)
                r = macro_f1(lab, pre[idx])
                self.ucb.update(dep, r)
                info['reward'] = dict(arch=dep, value=r, metric='macro_f1', timing='pre_update_deployed_arch')
                a = self.ucb.select(dinfo.get('severity', 0.0))
                info['ucb'] = dict(values=self.ucb.last_values, severity=dinfo.get('severity', 0.0), selected=a,
                                   n=[float(v) for v in self.ucb.n], r_hat=[float(v) for v in self.ucb.r_hat])
            else:
                a = self.ucb.select(True)
            self.arch = self.archs[a]
            if self.replay:
                tx = np.vstack([self.pool_X, self.ex_X]); ty = np.concatenate([self.pool_y, self.ex_y])
            else:
                tx, ty = X[idx], lab
            if self.distill and self.window_idx >= self.warmup_windows:
                self._train(self.teacher, tx, ty, self.teacher_epochs, 1e-4)
                self.model = self._train(self._new(self.arch), tx, ty, self.train_epochs, 1e-3, teacher=self.teacher)
                info.update(teacher_updated=True, distilled=True)
            else:
                self.model = self._train(self._new(self.arch), tx, ty, self.train_epochs, 1e-3)
            self._exemplar(X[idx], lab)
            if self.ucb_version == 'v1':
                r = binary_f1(lab, self.predict(X[idx]))  # R_t on Q_t only
                self.ucb.update(a, r)
                info.update(reward=dict(arch=a, value=r))
            info.update(queried=idx, train_size=len(ty))
            post = self.predict(X)
        else:
            post = pre.copy()
        info['next_arch'] = list(self.arch)
        self.window_idx += 1
        return pre, post, info

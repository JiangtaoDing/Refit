"""ReFIT strict-59 runner (see refit/core.py for what is kept / changed from 59).

Uses the cached streams built by refit.data.prepare(). Per window it stores
pre-update predictions, post-update predictions and the queried indices; metrics are
recorded for pre (frozen), post (full window) and post_unq (unqueried remainder =
within-window protocol). Counts are attack-positive; benign-positive is derived from them.
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from refit.core import Learner59, HybridVote59, Schedule
from refit.model import MLPModel
from refit.data import prepare, digest, dump, from_counts
ARCHS = [[128, 64], [32, 96], [96, 48], [64, 32]]
VERSION = 'refit-strict59-v1'

def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def counts(y, p):
    return dict(tp=int(((y == 1) & (p == 1)).sum()), fp=int(((y == 0) & (p == 1)).sum()),
                tn=int(((y == 0) & (p == 0)).sum()), fn=int(((y == 1) & (p == 0)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['2017', '2023', '2025'], required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--variant', choices=['full', 'periodic', 'random_trigger', 'random_sampling', 'fixed_retrain',
                                               'no_replay', 'no_distill', 'fixed_arch'], default='full')
    ap.add_argument('--cache', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--full-result', type=Path); ap.add_argument('--smoke', action='store_true')
    # sensitivity overrides (defaults = manuscript configuration)
    ap.add_argument('--quantile', type=float, default=.9); ap.add_argument('--vote-min', type=int, default=2)
    ap.add_argument('--eta', type=float, default=.6); ap.add_argument('--budget', type=float, default=.01)
    ap.add_argument('--rho', type=float, default=.75); ap.add_argument('--ucb-beta', type=float, default=.5)
    ap.add_argument('--ucb', choices=['v1', 'v2'], default='v2')  # ReFIT v2.0; v1 = strict-59 v1 reproduction
    ap.add_argument('--probe', type=int, default=0,
                    help='forgetting probe: keep N random labelled samples of W0 and of every window; after each update '
                         '(and at the end) record the current model on all probes seen so far. Evaluation only; uses its '
                         'own RNG, so the training trajectory is unchanged.')
    ap.add_argument('--gpu-data', action='store_true',
                    help='overhead timing: keep training tensors on the GPU (same batches and RNG draws as DataLoader)')
    ap.add_argument('--data-root', type=Path, default=Path('data'))
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a = ap.parse_args(); torch.set_num_threads(4)
    if not 0 < a.budget <= 1:
        ap.error('--budget must be in (0, 1]')
    if a.variant in ['periodic', 'random_trigger'] and a.full_result is None:
        ap.error('--full-result is required for matched-trigger variants')
    device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available() else ('cpu' if a.device == 'auto' else a.device))
    if device.type == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA is unavailable; use --device cpu')
    folder = prepare(a.dataset, a.cache, a.data_root)
    man = json.loads((folder / 'manifest.json').read_text())
    X = np.load(folder / 'X.npy', mmap_mode='r'); Y = np.load(folder / 'y.npy', mmap_mode='r')
    wins = man['windows'][:4] if a.smoke else man['windows']
    ep, tep = (1, 1) if a.smoke else (20, 10)
    a.output.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    schedule = None
    if a.variant in ['periodic', 'random_trigger']:
        full = json.loads(a.full_result.read_text()); trig = [r['window'] for r in full['rows'] if r['n_queried']]
        groups = {}
        for w in wins[1:]: groups.setdefault(math.ceil(.01 * w['n']), []).append(w['window'])
        schedule = []; rng = np.random.default_rng(a.seed + 2026)
        for _, ids in groups.items():
            c = sum(i in trig for i in ids)
            if c:
                ch = rng.choice(ids, c, replace=False) if a.variant == 'random_trigger' else np.asarray(ids)[np.round(np.linspace(0, len(ids) - 1, c)).astype(int)]
                schedule.extend(map(int, ch))
        det = Schedule(schedule)
    else:
        det = HybridVote59(X.shape[1], device=device, train_epochs=ep, quantile_alpha=a.quantile, vote_min=a.vote_min)
    # fixed_retrain = one arch + no distillation (original 59 ablation); R1.6 adds one component at a time:
    # fixed_arch = one arch (UCB off), no_distill = plain CE retraining, no_replay = train on Q_t only.
    fixed = a.variant in ('fixed_retrain', 'fixed_arch'); distill = a.variant not in ('fixed_retrain', 'no_distill')
    replay = a.variant != 'no_replay'
    L = Learner59(X.shape[1], ARCHS[:1] if fixed else ARCHS, MLPModel, det, device=device, train_epochs=ep, teacher_epochs=tep,
                  warmup_windows=10, exemplar_size=2000, anomaly_weight=a.eta, rho=a.rho, tau=3.,
                  selection='random' if a.variant == 'random_sampling' else 'mixed', distill=distill, ucb_version=a.ucb,
                  replay=replay, gpu_data=a.gpu_data)
    L.ucb.beta = a.ucb_beta
    init = wins[0]; ix = np.array(X[init['start']:init['end']]); iy = np.array(Y[init['start']:init['end']])
    if a.smoke: ix, iy = ix[:4096], iy[:4096]
    cfg = dict(version=VERSION if a.ucb == 'v1' else VERSION + '+ucb-v2', ucb=a.ucb, dataset=a.dataset, seed=a.seed, variant=a.variant, smoke=a.smoke, epochs=ep, teacher_epochs=tep,
               hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU', torch=torch.__version__, schedule=schedule,
               initialization_labels=len(iy), annotation=f'ceil({a.budget:g}*|W_t|) per triggered window', quantile=a.quantile, vote_min=a.vote_min, eta=a.eta, budget=a.budget, rho=a.rho, ucb_beta=a.ucb_beta, tau=3.,
               training_set='cumulative pool (all W0 + all queried) + exemplar 2000', ucb_reward='F1 of recovered student on Q_t only' if a.ucb == 'v1' else 'macro-F1 of deployed architecture pre-update predictions on Q_t (delayed); UCB1 alpha=1, (1+beta*severity) exploration',
               metrics='pre (frozen), post (full window), post_unq (unqueried remainder); attack-positive counts stored',
               archs=ARCHS[:1] if fixed else ARCHS, distill=distill, replay=replay, probe=a.probe, gpu_data=a.gpu_data,
               cache_manifest_sha256=digest(folder / 'manifest.json'))
    dump(a.output / 'config.json', cfg)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    L.initialize(ix, iy); rows = []; t_start = time.time()
    probes, prng = [], np.random.default_rng(a.seed + 99)

    def add_probe(wid, px, py):
        if a.probe:
            i = np.sort(prng.choice(len(py), min(a.probe, len(py)), replace=False)); probes.append((wid, px[i].copy(), py[i].copy()))

    def probe_eval():
        return [dict(window=wid, **counts(py, L.predict(px))) for wid, px, py in probes]
    add_probe(init['window'], ix, iy); probe_init = probe_eval() if a.probe else None
    for w in wins[1:]:
        x = np.array(X[w['start']:w['end']]); y = np.array(Y[w['start']:w['end']])
        if a.smoke: x, y = x[:4096], y[:4096]
        L.label_budget = math.ceil(a.budget * len(x)); asked = []

        def oracle(idx):
            if asked: raise RuntimeError('repeated oracle call')
            asked.append(1); return y[idx].copy()
        synchronize(device); t0 = time.time()
        pre, post, info = L.recover_window(x, oracle)
        synchronize(device); sec = time.time() - t0
        q = info['queried']; keep = np.ones(len(y), bool); keep[q] = False
        add_probe(w['window'], x, y)
        probe = probe_eval() if a.probe and (len(q) or w is wins[-1]) else None
        row = dict(window=w['window'], n_samples=len(y), n_queried=len(q), trigger=info['trigger'], votes=info['votes'],
                   teacher_updated=info['teacher_updated'], distilled=info['distilled'], deployed_arch=info['deployed_arch'],
                   next_arch=info['next_arch'], reward=info['reward'], ucb=info.get('ucb'), train_size=info['train_size'], seconds=sec, probe=probe,
                   pre=counts(y, pre), post=counts(y, post), post_unq=counts(y[keep], post[keep]))
        rows.append(row)
        np.savez_compressed(a.output / f"window_{w['window']:03d}.npz", y_true=y, y_pred_pre=pre, y_pred_post=post, queried_indices=q)
        with (a.output / 'events.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
        print(json.dumps(dict(ds=a.dataset, seed=a.seed, v=a.variant, w=w['window'], q=len(q), sec=round(sec, 1))), flush=True)
    summ = dict(total_queries=sum(r['n_queried'] for r in rows), updates=sum(r['n_queried'] > 0 for r in rows), seconds=time.time() - t_start)
    for k in ['pre', 'post', 'post_unq']:
        summ[k + '_attack_window_mean'] = {m: float(np.mean([from_counts(**r[k])[m] for r in rows])) for m in ['precision', 'recall', 'f1']}
        summ[k + '_benign_window_mean'] = {m: float(np.mean([from_counts(tp=r[k]['tn'], fp=r[k]['fn'], tn=r[k]['tp'], fn=r[k]['fp'])[m] for r in rows])) for m in ['precision', 'recall', 'f1']}
    if schedule is not None:
        assert summ['total_queries'] == full['summary']['total_queries'] and summ['updates'] == full['summary']['updates']
    dump(a.output / 'result.json', dict(config=cfg, summary=summ, rows=rows, probe_init=probe_init))
    print('COMPLETE', a.output, flush=True)


if __name__ == '__main__':
    main()

"""Multi-host pool for the strict-59 sensitivity sweep (Fig. 4): one hyper-parameter varied at a time,
all others at the manuscript default; 3 datasets x 3 seeds per value. The default value of every
panel is the existing Full run (refit59/runs/<ds>_seed<s>_full; with --ucb v2 use
refit59_ucbv2/runs/<ds>_seed<s>_full) and is not rerun.

Jobs are claimed atomically with mkdir(claims/<id>) so several hosts can share one batch dir.
Usage: python refit59_sens_pool.py --batch DIR --cache DIR [--workers 9] [--ucb v1]   (default v2)
"""
import argparse, os, socket, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRID = {  # flag: (values, default)
    'quantile': ([.80, .85, .90, .95, .99], .90),
    'vote-min': ([1, 2, 3], 2),
    'eta': ([0, .2, .4, .6, .8, 1], .6),
    'budget': ([.0025, .005, .01, .02, .05], .01),
    'rho': ([0, .25, .5, .75, 1], .75),
    'ucb-beta': ([0, .5, 1, 2], .5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=Path, required=True); ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=9)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--ucb', choices=['v1', 'v2'], default='v2')
    a = ap.parse_args()
    batch = a.batch.resolve(); runs, logs, claims = batch / 'runs', batch / 'logs', batch / 'claims'
    for d in (runs, logs, claims): d.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    jobs = []
    for ds in ['2017', '2023', '2025']:          # 2017 (slowest) first
        for flag, (vals, dflt) in GRID.items():
            for v in vals:
                if v == dflt: continue
                for s in a.seeds:
                    jid = f'{ds}_seed{s}_{flag}{v:g}'
                    cmd = [sys.executable, str(HERE / 'run.py'), '--dataset', ds, '--seed', str(s), '--variant', 'full',
                           '--cache', str(a.cache.resolve()), '--output', str(runs / jid), f'--{flag}', f'{v:g}', '--ucb', a.ucb]
                    jobs.append((jid, runs / jid, cmd))

    def run(job):
        jid, out, cmd = job
        if (out / 'result.json').exists(): return
        try: (claims / jid).mkdir()
        except FileExistsError: return
        (claims / jid / 'host').write_text(socket.gethostname())
        with open(logs / f'{jid}.log', 'wb') as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(HERE.parent))
        (claims / jid / 'rc').write_text(str(rc))

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))
    done = sum((o / 'result.json').exists() for _, o, _ in jobs)
    print('HOST_DONE', socket.gethostname(), done, '/', len(jobs), flush=True)


if __name__ == '__main__':
    main()

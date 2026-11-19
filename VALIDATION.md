# ReFIT v2.0 packaging validation

Source: the author-designated local `IoT_NIDS/REFIT_V2.0.md` and its current code files. Validation environment: Windows, Python 3.11.5, PyTorch 2.1.1+cpu, NumPy 1.24.3, pandas 2.1.1 and scikit-learn 1.3.0. This is not the original A6000 experiment environment.

Passed:

- Byte-for-byte equality of `refit/core.py` with local `IoT_NIDS/src/refit59/learner59.py`; MD5 matches the version document.
- The nine original UCB and R1.6 tests (only imports adapted), plus two packaging tests: **11 tests passed**.
- Original test source MD5 matches the version document; extracted MLP class matches the local source structurally.
- Both CLI argument definitions default to v2. Direct learner default remains v1 as in the authoritative source.
- Synthetic CPU runs without `--ucb`: v2 selected, initialization and two recovery windows completed, query budgets and pre-update rewards checked, forgetting-probe records saved.
- Same synthetic input and seed with DataLoader and tensor-resident training produced identical saved labels, queries, pre-update and post-update predictions on CPU. This is not CUDA timing validation.
- Both CLI help commands, Python parsing, citation YAML parsing and ZIP integrity checks passed.

Upstream scikit-learn warnings about `n_init` defaults and Windows/MKL threading were observed. Clustering settings were preserved. The full sensitivity sweep, real datasets, three-seed paper tables, CUDA behavior and server deployment were not rerun.

Functional checks and source integrity do not certify correspondence to every manuscript equation. The existing alignment notes are retained in `RELEASE_NOTES.md`; no algorithm was altered to resolve them.

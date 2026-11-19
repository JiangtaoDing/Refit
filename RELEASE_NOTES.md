# ReFIT v2.0 source and packaging

Release source: the current local `IoT_NIDS` files identified by `IoT_NIDS/REFIT_V2.0.md`, as explicitly selected by the author. This replaces the earlier WSL-based extraction. `SOURCE_MANIFEST.json` records the authority document hash and source/package hashes.

- `refit/core.py` is byte-identical to local `src/refit59/learner59.py`; MD5 `8e3669ce53b84617c582fb5390adad38` matches the version document.
- CLI defaults remain v2 in `scripts/run.py` (source `refit59_run.py`) and `scripts/refit59_sens_pool.py`.
- Direct `Learner59(...)` construction retains the original default v1. Python callers must specify `ucb_version="v2"` for ReFIT v2.0.
- The complete local runner includes R1.6 variants (`no_replay`, `no_distill`, `fixed_arch`), `--probe`, and `--gpu-data`.
- The nine original tests are included with import-only changes; the MLP class is unchanged, with unused training helpers omitted.
- Packaging adaptations are limited to module paths, the sensitivity runner filename, configurable input roots, CPU/automatic device selection, conditional CUDA synchronization and CLI/cache checks. The core algorithm and its historical module header are preserved exactly; that header describes legacy behavior as well as shared components, so use the v2 branch and this documentation to interpret current reward behavior.

The package excludes datasets, trained weights, result tables, baseline code, manuscript files, server-specific shell launchers and upstream raw-data construction. The sensitivity launcher remains because it is small and directly invokes the core runner. It expects prepared caches. The minimal runner reports per-class positive-label metrics; the paper's support-weighted main-table aggregation is separate.

Source hashes establish local extraction integrity, not a new reproduction of every experiment. Complete datasets and A6000 runs were not executed during this packaging task.

## Subsequent manuscript alignment audit

The earlier audit concerns the same HISS/DART logic retained in this local release source. Compared against the active, uncommented text in the local revised manuscript's architecture and evaluation sections. The main TeX file includes these sections. No algorithm or manuscript changes were made in this audit.

| Area | Manuscript | Packaged implementation | Assessment |
| --- | --- | --- | --- |
| HISS rarity | Inverse cluster size, normalized and combined with entropy | Cluster-dependent random preselection, resized to exactly K samples; binary membership indicator combined with entropy | Material algorithm mismatch |
| DART history | Sliding and quantile criteria use the same previous-k-window history; k=5 | Separate buffers of size 5 and 20 include the current window before it is excluded, leaving at most 4 and 19 previous windows | Material trigger mismatch |
| DART warm-up | Sliding criterion requires at least 3 previous windows | Enabled with 3 buffer entries including the current one, hence 2 previous windows; quantile enabled with 4 previous windows | Off-by-one / unspecified quantile warm-up |
| MAD handling | Divide by MAD + epsilon | Disable fixed/sliding alarm when MAD < 1e-6; otherwise divide by MAD without epsilon | Material difference for stable histories |
| UCB reward timing and exploration | Reward deployed architecture on frozen pre-update queried predictions, then use drift-scaled UCB1 | Matches these mechanisms | Aligned at mechanism/formula level |
| Macro-F1 class set | Text says classes present in Q_t | Union of classes present in true labels OR predictions | Definition needs clarification |
| Recovery and distillation | Cumulative pool plus replay, window-index warm-up at 10, teacher update then fresh student with CE/KL | Matches inspected code and defaults | Aligned; replay is concatenation with duplicate weighting |
| Main-table metrics | Class-support-weighted P/R/F1 | Runner summary provides attack-positive and benign-positive scores separately | Missing main-table aggregation in the minimal package; counts permit recomputation |

Source anchors in the local revised manuscript: architecture lines 68, 80–97, 138–140, 192–234, 404–448 and 450–516; evaluation lines 25 and 292. Code anchors: `refit/core.py` detector, `_select`, `macro_f1` and `recover_window`; `scripts/run.py` summary construction.

The HISS discrepancy has a stronger consequence at the default eta=0.6: the K preselected points receive scores approximately in [0.6, 1], while all other points receive scores in [0, 0.4]. Thus the final top-K set is the preselection itself; entropy cannot change membership. A controlled 200-sample diagnostic with cluster sizes 10 and 190, K=10, selected only 2 samples from the small cluster, and reversing entropy rankings did not change the selected set. The manuscript's inverse-size rule selects all 10 small-cluster points in that example.

Additional controlled diagnostics confirmed a sliding alarm after only two previous windows, actual history lengths of 4 and 19, and a disabled fixed alarm for reference MAD=0 despite a large median increase. For true labels [0,0] and predictions [0,1], the code's macro-F1 is 1/3; averaging only over the true-label class gives 2/3.

The user-designated release source is now the local ReFIT v2.0 tree. Its version document records A6000 provenance; the core MD5 and original test MD5 match that document. No remote provenance re-audit was performed. The identified manuscript differences were not silently changed during packaging. Algorithm changes intended to match manuscript equations require renewed validation rather than reuse of old numbers.

# ReFIT v2.0 — core implementation

Research code for **ReFIT: A Drift-Aware Recovery Framework for Resource-Efficient IoT Intrusion Detection**.

## Included code

| File | Purpose |
| --- | --- |
| `refit/core.py` | Hybrid-vote drift detector, mixed query sampling, DA-UCB, cumulative labeled replay, teacher update and student distillation |
| `refit/model.py` | MLP backbone |
| `refit/data.py` | Three-dataset stream preparation and initialization-only scaling |
| `scripts/run.py` | Full method, seven ablations, forgetting probes and GPU tensor batching; saved pre/post predictions |
| `scripts/refit59_sens_pool.py` | Original sensitivity sweep; v2 by default; uses prepared caches |
| `tests/test_refit59_ucb.py` | Nine original UCB and component-ablation tests, with adapted imports |
| `tests/test_core.py` | Two CPU checks of querying, recovery, distillation and detection |

Baselines, datasets, trained weights, experimental output, manuscript files and machine-specific experiment shell scripts are excluded. Source hashes are listed in `SOURCE_MANIFEST.json`. Historical `59` class names and the small legacy UCB selector are retained to make the extraction traceable; `--ucb v1` explicitly enables the older reward policy.

## Install and check

The manuscript experiment environment uses Python 3.12, PyTorch 2.6 and scikit-learn 1.7.2. Local packaging checks use Python 3.11 and CPU PyTorch; see `VALIDATION.md`. Create a virtual environment, activate it, then run from this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/run.py --help
```

For GPU experiments, install the appropriate CUDA-enabled PyTorch build for your system. `--device auto` selects CUDA when available, otherwise CPU. CPU support is for portability; it does not establish equivalence to the GPU paper runs. Dependency ranges are installation constraints, not the original experiment lockfile.

## Data layout

Prepare the **same ordered, preprocessed CSV streams** expected by the experiment, then arrange them as follows:

```text
data/
  2017/CIC_17_day1.csv ... CIC_17_day5.csv
  2023/train.csv
  2023/test.csv
  2025/Main_CIC_IIoT2025_forced_drift_binary_light_ref_scaled.csv
```

Pass another parent directory with `--data-root`. Input files and cache arrays are not distributed.

| Dataset ID | Expected source contract |
| --- | --- |
| `2017` | Five ordered CSV files with a `Label` column (`benign` vs. attack) and numeric features. Drop invalid rows; use the first 2,800,000 valid rows. The first 500,000 must be benign initialization samples; remaining windows contain 50,000 rows. |
| `2023` | Numeric features plus `y_binary`; optional `y_class`, `y_family` are excluded. Initialization selects the first 25,000 rows of each class from the predeployment train pool, restoring source order. Test stream: 1,800,000 rows, in 50,000-row windows. |
| `2025` | Numeric features, binary `label`, contiguous `window_id` groups; 227,191 rows and 23 windows. The first 10,000 rows contain 500 attacks. |

Labels are `0 = benign`, `1 = attack`. A StandardScaler is fitted on the initialization window only. The resulting cache contains `X.npy`, `y.npy`, `scaler.npz` and `manifest.json`; the manifest records feature order, windows and input hashes. Use a fresh cache directory when changing source files. Existing complete caches are reused.

These CSVs are project-specific prepared inputs, not interchangeable with arbitrary raw publisher downloads. The upstream construction of the 2023 novel-drift split and 2025 forced-drift stream is outside this minimal package. Full dataset reconstruction and numerical reproduction require those preprocessing assets and the exact inputs.

## Run

```bash
python scripts/run.py --dataset 2017 --seed 1 --data-root data --cache cache --output outputs/2017_seed1_full --device auto
```

Change `--dataset` to `2023` or `2025`. Run seeds `1`, `2`, `3` separately. Output directories must not already exist.

The default variant is `full`. Other variants are `random_sampling`, `fixed_retrain`, `periodic`, `random_trigger`, `no_replay`, `no_distill`, and `fixed_arch`. The two matched-trigger variants require the completed full-method result for the same dataset and seed:

```bash
python scripts/run.py --dataset 2017 --seed 1 --variant periodic --full-result outputs/2017_seed1_full/result.json --cache cache --output outputs/2017_seed1_periodic
```

Keep the same budget and smoke/full setting between these paired runs; the historical matching schedule assumes the default 1% budget. `--smoke` reduces epochs and takes at most 4,096 samples from each of the first four windows, but still requires the real prepared input/cache. The unittest checks use generated data and require no datasets.

Use `--probe 2000` to record evaluation-only forgetting probes. `--gpu-data` enables the source implementation's tensor-resident training path, intended for GPU overhead measurement. The sensitivity launcher uses existing caches: `python scripts/refit59_sens_pool.py --batch outputs/sensitivity --cache cache --workers 1 --seeds 1`. It does not rerun default-grid configurations; retain matching full runs separately. Add `--ucb v1` to either entry point only when reproducing legacy v1 runs.

Defaults: 20 student epochs, 10 teacher epochs, warm-up window 10, exemplar size 2,000, query budget `ceil(0.01 * window_size)`, anomaly weight 0.6, distillation weight 0.75, temperature 3, and architectures `[128,64]`, `[32,96]`, `[96,48]`, `[64,32]`.

## Evaluation protocol

Each output includes `config.json`, `events.jsonl`, `result.json` and per-window prediction NPZ files. The runner does not export a final model checkpoint.

- `pre`: frozen predictions before querying or training on the current window.
- `post`: predictions after recovery, evaluated over the full current window, including queried samples.
- `post_unq`: predictions after recovery, evaluated only on the unqueried remainder.

Full-window labels are available only to the offline evaluator; the learner receives selected labels through `oracle(indices)`. UCB v2 uses delayed macro-F1 from the deployed model's **pre-update** predictions on queried samples, before recovery training. Macro-F1 averages over classes present in the true labels or predictions. Untriggered windows provide no reward. Drift severity scales the exploration bonus. The optional legacy v1 instead rewards post-update attack-positive F1 on the queried training samples.

Confusion counts use attack as the positive class. Run summaries provide both attack-positive and benign-positive precision/recall/F1, with equal weight per online window. These are distinct from class-weighted metrics and pooled confusion-count metrics. Across seeds, use the mean and sample standard deviation. Never label post-update values as prequential performance.

## Citation

Cite version 2.0 of the software using GitHub's **Cite this repository** button or the metadata in [CITATION.cff](CITATION.cff). The [v2.0 release](https://github.com/JiangtaoDing/Refit/releases/tag/v2.0) identifies the source used for this version.

## License

ReFIT is released under the [MIT License](LICENSE).

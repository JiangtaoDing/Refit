# ReFIT v2.0

ReFIT is a drift-aware recovery framework for IoT intrusion detection. It monitors incoming windows of network traffic for changes in the data distribution. When a window triggers recovery, ReFIT selects traffic samples for labeling and updates the detector using queried labels, replay, and teacher-guided distillation. The default label budget is 1% of each triggered window, rounded up to the next whole sample.

## Installation

Clone the repository and install the dependencies in a Python 3.12 environment.

```bash
git clone https://github.com/JiangtaoDing/Refit.git
cd Refit
python -m pip install -r requirements.txt
```

For GPU execution, use a CUDA-enabled build of PyTorch. The runner selects CUDA when available and otherwise uses the CPU.

## Running ReFIT

Place the prepared CSV stream for your dataset under `data/`. The runner expects the following files and window order.

| Dataset | Files | Stream |
| --- | --- | --- |
| CIC-IDS2017 | `2017/CIC_17_day1.csv` through `2017/CIC_17_day5.csv` | Five files in day order. The first 500,000 valid rows are benign initialization samples, followed by 46 windows of 50,000 rows. |
| CIC-IoT2023 | `2023/train.csv`, `2023/test.csv` | Initialization uses the first 25,000 training rows per class. The prepared test stream has 36 windows of 50,000 rows. |
| CIC-IIoT2025 | `2025/Main_CIC_IIoT2025_forced_drift_binary_light_ref_scaled.csv` | The prepared stream has 227,191 rows in 23 contiguous `window_id` groups, starting with the initialization window. |

Run the full method on CIC-IDS2017 with seed 1.

```bash
python scripts/run.py --dataset 2017 --seed 1 --data-root data --cache cache --output outputs/2017_seed1_full
```

Use `--dataset 2023` or `--dataset 2025` for the other streams. Choose a new output directory for each run. The runner caches the prepared data, so use a fresh cache directory if you change the input files.

The output directory contains `result.json`, per-window predictions, and the indices of queried samples. Use `pre` for online predictions made before querying labels. The `post` scores cover the entire window after recovery, and `post_unq` covers only its unqueried samples. Run `python scripts/run.py --help` for ablations and other options.

## Citation

To cite ReFIT in a paper, use the [v2.0 release](https://github.com/JiangtaoDing/Refit/releases/tag/v2.0). Select **Cite this repository** on GitHub to copy a BibTeX entry, or use the author and version details in [CITATION.cff](CITATION.cff).

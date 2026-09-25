<<<<<<< HEAD
<<<<<<< HEAD
# VANET Normal / Sybil / Illusion detector

This project is a complete, reproducible PyTorch pipeline for classifying vehicle-to-vehicle messages as **Normal**, **Sybil**, or **Illusion**. It downloads checksum-verified VeReMi NextGen data, preserves the official splits, creates leakage-resistant temporal and same-receiver relationship features, trains a two-branch TCN–BiGRU–attention network, calibrates its probabilities, evaluates every class, and exports deployable predictions.

Start with [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md) if these terms are new. See [DATASETS.md](DATASETS.md) for the exact datasets and [RSSI_INTEGRATION.md](RSSI_INTEGRATION.md) to add receiver signal strength correctly.

## What is included

- Official-dataset downloader with size and MD5 verification
- Automatic retry/backoff for temporary Zenodo rate limits and server timeouts
- Streaming parser for VeReMi's nested ZIP archives
- Normal, Sybil, and three position-based Illusion sources
- Cold-start handling for short-lived Sybil aliases
- Optional measured RSSI with missing-value masks and safety checks
- TCN + BiGRU + attention behavior encoder
- BiGRU + attention relationship encoder
- Focal loss, class weights, auxiliary attack-specific heads, AdamW, gradient clipping, mixed precision, scheduler, and early stopping
- Validation-only temperature scaling and threshold selection
- Automated validation-only tuning of learning rate, dropout, sequence length, and imbalance strategy
- Optional hierarchical Normal-vs-Attack then Sybil-vs-Illusion classifier
- Multi-seed final reporting with mean and standard deviation
- Per-class metrics, confusion matrices, plots, checkpoint, CSV inference, and a multi-message alert rule
- Automated unit tests and an official-data smoke-validation procedure

## Requirements

- Python 3.10 or newer
- At least 8 GB RAM; 16 GB is more comfortable for the recommended profile
- At least 20 GB free disk for the recommended download, overlapping sequence shards, and outputs; actual use depends on stride and selected scenarios
- An NVIDIA GPU is strongly recommended for full training, but CPU training works

## Installation

Open a terminal in this folder.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

### Linux, macOS, or Google Colab

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

In Colab, first select a GPU runtime, upload and unzip the project, then run the commands from the project directory. You can omit the virtual-environment commands in Colab and run `pip install -e .` directly.

## Recommended end-to-end run

### 1. Download real dataset archives

```bash
python download_data.py --profile recommended --output data/raw
```

`recommended` downloads the highway and urban two-lane scenarios: 10 files and approximately 0.96 GiB. Use `quick` for a smaller functionality run or `full` for all four scenarios and approximately 11.44 GiB. Each scenario includes baseline Normal traffic, traffic-congestion Sybil, constant-position-offset, random-position-offset, and position-mirroring archives.

### 2. Build model-ready sequences

```bash
python prepare_data.py \
  --raw-dir data/raw \
  --output-dir data/processed \
  --seq-len 16 \
  --stride 2 \
  --overwrite
```

Windows PowerShell accepts the same command on one line. The final printed manifest must show non-zero Normal, Sybil, and Illusion counts in Train, Validation, and Test.

### 3. Train and optimize

```bash
python train.py --config configs/recommended.yaml --run-dir runs/recommended
```

To restart and replace an existing run, add `--overwrite`. The best checkpoint is selected by validation macro-F1, then probability temperature and the attack alert threshold are fitted only on validation data.

### Automated optimization and multiple seeds

For the research run, use the optimizer to compare class weighting with balanced
sampling and tune learning rate, dropout, and sequence length. It selects only
with Validation Macro-F1, freezes the setup, and only then evaluates final seeds
on Test:

```bash
python optimize_experiments.py --profile recommended --raw-dir data/raw \
  --processed-root data/optimized --runs-dir runs/optimization \
  --seeds 42 43 44
```

The controlled search tests sequence lengths 16, 24, and 32; learning rates
0.001, 0.0005, and 0.0003; dropout 0.20, 0.25, and 0.30; and both imbalance
strategies. If Sybil or Illusion validation F1 remains below 0.65, a hierarchical
classifier is compared automatically.

Preview the experiments without training:

```bash
python optimize_experiments.py --plan-only
```

For the final all-scenario experiment, first download the full profile and then
use `--final-full`. Tuning uses the recommended two-scenario subset; the frozen
winning setup is retrained on all four scenarios:

```bash
python download_data.py --profile full --output data/raw
python optimize_experiments.py --profile recommended --final-full \
  --raw-dir data/raw --processed-root data/optimized \
  --runs-dir runs/full_optimization --seeds 42 43 44
```

See [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) for Windows, CPU, Colab,
output files, and restart instructions.

### 4. Review independent test results

The training command automatically creates these files under `runs/recommended/`:

- `best_model.pt`—model, scaler, configuration, calibration, and threshold
- `test_metrics.json`—overall and per-class numerical metrics
- `test_classification_report.json`—precision, recall, and F1 for every class
- `test_confusion_matrix.csv` and `.png`
- `training_history.csv` and `.png`

You can reproduce evaluation from the saved checkpoint:

```bash
python evaluate.py \
  --checkpoint runs/recommended/best_model.pt \
  --processed-dir data/processed \
  --output-dir runs/recommended/recheck
```

### 5. Predict new messages

Input can be one VeReMi-style receiver JSON file, a flat CSV, or a ZIP that directly contains receiver JSON files.

```bash
python predict.py \
  --checkpoint runs/recommended/best_model.pt \
  --input path/to/receiver_messages.json \
  --output predictions.csv
```

The CSV contains each class probability, malicious risk, prediction, smoothed risk, recent suspicious count, vehicle alert, and a conservative recommended action.

## Adding RSSI

VeReMi NextGen does not provide RSSI, so normal training uses an explicit “RSSI unavailable” mask. To use real or radio-simulator measurements, provide a CSV containing `receiver_id,message_id,rssi_dbm` during both preprocessing and deployment:

```bash
python prepare_data.py --raw-dir data/raw --output-dir data/processed-rssi \
  --rssi-csv data/rssi.csv --overwrite
```

Then change `data_dir` in a copied config to `data/processed-rssi`, train a new checkpoint, and pass compatible RSSI to `predict.py`. Full details and quality rules are in [RSSI_INTEGRATION.md](RSSI_INTEGRATION.md).

## Configuration choices

| File | Use |
|---|---|
| `configs/recommended.yaml` | Main GPU experiment, 50 epochs maximum |
| `configs/recommended_balanced.yaml` | Balanced-sampler comparison without duplicate class weighting |
| `configs/recommended_hierarchical.yaml` | Optional two-stage decision structure |
| `configs/full.yaml` | Direct all-four-scenario experiment after full preprocessing |
| `configs/quick.yaml` | Smaller architecture and shorter experiment |
| `configs/cpu_smoke.yaml` | Integration check only; not a benchmark |

If GPU memory is insufficient, first lower `training.batch_size`. Do not tune against the Test split. For a final report, run several seeds (for example 42, 43, and 44) and report the mean and standard deviation from untouched Test results.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

## Honest performance expectations

No architecture can guarantee a particular score before it is trained and tested on the final data. This repository deliberately does not hard-code or advertise unverified metrics. It is designed to improve overall class balance and reduce leakage, but real-road generalization must be verified on receiver data from the target deployment environment.

## Project layout

```text
configs/                 training configurations
src/vanet_detector/      downloader, parser, features, model, training, evaluation, inference
optimize_experiments.py  validation tuning, hierarchical comparison, and multi-seed reporting
tests/                   automated correctness tests
data/raw/                downloaded ZIP archives (not packaged)
data/processed/          generated NumPy shards (not packaged)
runs/                    trained models and reports (not packaged)
```

## Reproducibility

The default seed is 42 and deterministic PyTorch behavior is requested. GPU libraries can still introduce small platform-dependent differences. Keep the download manifest, preprocessing manifest, resolved training config, package versions, and final checkpoint with every reported experiment.
=======
# major-project-tcn-bigru-
>>>>>>> daeb26f749ff6eef23b4998222f8817cab36c158
=======
<<<<<<< HEAD
# VANET Normal / Sybil / Illusion detector

This project is a complete, reproducible PyTorch pipeline for classifying vehicle-to-vehicle messages as **Normal**, **Sybil**, or **Illusion**. It downloads checksum-verified VeReMi NextGen data, preserves the official splits, creates leakage-resistant temporal and same-receiver relationship features, trains a two-branch TCN–BiGRU–attention network, calibrates its probabilities, evaluates every class, and exports deployable predictions.

Start with [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md) if these terms are new. See [DATASETS.md](DATASETS.md) for the exact datasets and [RSSI_INTEGRATION.md](RSSI_INTEGRATION.md) to add receiver signal strength correctly.

## What is included

- Official-dataset downloader with size and MD5 verification
- Automatic retry/backoff for temporary Zenodo rate limits and server timeouts
- Streaming parser for VeReMi's nested ZIP archives
- Normal, Sybil, and three position-based Illusion sources
- Cold-start handling for short-lived Sybil aliases
- Optional measured RSSI with missing-value masks and safety checks
- TCN + BiGRU + attention behavior encoder
- BiGRU + attention relationship encoder
- Focal loss, class weights, auxiliary attack-specific heads, AdamW, gradient clipping, mixed precision, scheduler, and early stopping
- Validation-only temperature scaling and threshold selection
- Automated validation-only tuning of learning rate, dropout, sequence length, and imbalance strategy
- Optional hierarchical Normal-vs-Attack then Sybil-vs-Illusion classifier
- Multi-seed final reporting with mean and standard deviation
- Per-class metrics, confusion matrices, plots, checkpoint, CSV inference, and a multi-message alert rule
- Automated unit tests and an official-data smoke-validation procedure

## Requirements

- Python 3.10 or newer
- At least 8 GB RAM; 16 GB is more comfortable for the recommended profile
- At least 20 GB free disk for the recommended download, overlapping sequence shards, and outputs; actual use depends on stride and selected scenarios
- An NVIDIA GPU is strongly recommended for full training, but CPU training works

## Installation

Open a terminal in this folder.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

### Linux, macOS, or Google Colab

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

In Colab, first select a GPU runtime, upload and unzip the project, then run the commands from the project directory. You can omit the virtual-environment commands in Colab and run `pip install -e .` directly.

## Recommended end-to-end run

### 1. Download real dataset archives

```bash
python download_data.py --profile recommended --output data/raw
```

`recommended` downloads the highway and urban two-lane scenarios: 10 files and approximately 0.96 GiB. Use `quick` for a smaller functionality run or `full` for all four scenarios and approximately 11.44 GiB. Each scenario includes baseline Normal traffic, traffic-congestion Sybil, constant-position-offset, random-position-offset, and position-mirroring archives.

### 2. Build model-ready sequences

```bash
python prepare_data.py \
  --raw-dir data/raw \
  --output-dir data/processed \
  --seq-len 16 \
  --stride 2 \
  --overwrite
```

Windows PowerShell accepts the same command on one line. The final printed manifest must show non-zero Normal, Sybil, and Illusion counts in Train, Validation, and Test.

### 3. Train and optimize

```bash
python train.py --config configs/recommended.yaml --run-dir runs/recommended
```

To restart and replace an existing run, add `--overwrite`. The best checkpoint is selected by validation macro-F1, then probability temperature and the attack alert threshold are fitted only on validation data.

### Automated optimization and multiple seeds

For the research run, use the optimizer to compare class weighting with balanced
sampling and tune learning rate, dropout, and sequence length. It selects only
with Validation Macro-F1, freezes the setup, and only then evaluates final seeds
on Test:

```bash
python optimize_experiments.py --profile recommended --raw-dir data/raw \
  --processed-root data/optimized --runs-dir runs/optimization \
  --seeds 42 43 44
```

The controlled search tests sequence lengths 16, 24, and 32; learning rates
0.001, 0.0005, and 0.0003; dropout 0.20, 0.25, and 0.30; and both imbalance
strategies. If Sybil or Illusion validation F1 remains below 0.65, a hierarchical
classifier is compared automatically.

Preview the experiments without training:

```bash
python optimize_experiments.py --plan-only
```

For the final all-scenario experiment, first download the full profile and then
use `--final-full`. Tuning uses the recommended two-scenario subset; the frozen
winning setup is retrained on all four scenarios:

```bash
python download_data.py --profile full --output data/raw
python optimize_experiments.py --profile recommended --final-full \
  --raw-dir data/raw --processed-root data/optimized \
  --runs-dir runs/full_optimization --seeds 42 43 44
```

See [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) for Windows, CPU, Colab,
output files, and restart instructions.

### 4. Review independent test results

The training command automatically creates these files under `runs/recommended/`:

- `best_model.pt`—model, scaler, configuration, calibration, and threshold
- `test_metrics.json`—overall and per-class numerical metrics
- `test_classification_report.json`—precision, recall, and F1 for every class
- `test_confusion_matrix.csv` and `.png`
- `training_history.csv` and `.png`

You can reproduce evaluation from the saved checkpoint:

```bash
python evaluate.py \
  --checkpoint runs/recommended/best_model.pt \
  --processed-dir data/processed \
  --output-dir runs/recommended/recheck
```

### 5. Predict new messages

Input can be one VeReMi-style receiver JSON file, a flat CSV, or a ZIP that directly contains receiver JSON files.

```bash
python predict.py \
  --checkpoint runs/recommended/best_model.pt \
  --input path/to/receiver_messages.json \
  --output predictions.csv
```

The CSV contains each class probability, malicious risk, prediction, smoothed risk, recent suspicious count, vehicle alert, and a conservative recommended action.

## Adding RSSI

VeReMi NextGen does not provide RSSI, so normal training uses an explicit “RSSI unavailable” mask. To use real or radio-simulator measurements, provide a CSV containing `receiver_id,message_id,rssi_dbm` during both preprocessing and deployment:

```bash
python prepare_data.py --raw-dir data/raw --output-dir data/processed-rssi \
  --rssi-csv data/rssi.csv --overwrite
```

Then change `data_dir` in a copied config to `data/processed-rssi`, train a new checkpoint, and pass compatible RSSI to `predict.py`. Full details and quality rules are in [RSSI_INTEGRATION.md](RSSI_INTEGRATION.md).

## Configuration choices

| File | Use |
|---|---|
| `configs/recommended.yaml` | Main GPU experiment, 50 epochs maximum |
| `configs/recommended_balanced.yaml` | Balanced-sampler comparison without duplicate class weighting |
| `configs/recommended_hierarchical.yaml` | Optional two-stage decision structure |
| `configs/full.yaml` | Direct all-four-scenario experiment after full preprocessing |
| `configs/quick.yaml` | Smaller architecture and shorter experiment |
| `configs/cpu_smoke.yaml` | Integration check only; not a benchmark |

If GPU memory is insufficient, first lower `training.batch_size`. Do not tune against the Test split. For a final report, run several seeds (for example 42, 43, and 44) and report the mean and standard deviation from untouched Test results.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

## Honest performance expectations

No architecture can guarantee a particular score before it is trained and tested on the final data. This repository deliberately does not hard-code or advertise unverified metrics. It is designed to improve overall class balance and reduce leakage, but real-road generalization must be verified on receiver data from the target deployment environment.

## Project layout

```text
configs/                 training configurations
src/vanet_detector/      downloader, parser, features, model, training, evaluation, inference
optimize_experiments.py  validation tuning, hierarchical comparison, and multi-seed reporting
tests/                   automated correctness tests
data/raw/                downloaded ZIP archives (not packaged)
data/processed/          generated NumPy shards (not packaged)
runs/                    trained models and reports (not packaged)
```

## Reproducibility

The default seed is 42 and deterministic PyTorch behavior is requested. GPU libraries can still introduce small platform-dependent differences. Keep the download manifest, preprocessing manifest, resolved training config, package versions, and final checkpoint with every reported experiment.
=======
# major-project-tcn-bigru-
>>>>>>> daeb26f749ff6eef23b4998222f8817cab36c158
>>>>>>> 6bd5b795979070b4df9b8c7c50f3dc86e29a7937

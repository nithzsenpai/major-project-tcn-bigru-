# Optimization and final-training guide

This workflow implements the complete improvement plan without using the Test
split to choose hyperparameters.

## What the optimizer does

1. Verifies that every required archive is present.
2. Includes baseline Normal, traffic-congestion Sybil, constant position
   offset, random position offset, and position mirroring for every scenario.
3. Builds separate sequence datasets for lengths 16, 24, and 32 with stride 2.
4. Runs a controlled eight-trial search on the first seed.
5. Compares training-only class weighting with training-only balanced sampling.
6. Selects by Validation Macro-F1. Ties prefer Illusion F1, balanced accuracy,
   and then lower false-positive rate.
7. If either attack-class F1 is below 0.65, compares the hierarchical model.
8. Retrains the frozen winning configuration with seeds 42, 43, and 44.
9. Evaluates Test only after the model choices are frozen.
10. Reports mean, standard deviation, minimum, and maximum final metrics.

The default model can train for at most 50 epochs, but early stopping can finish
it earlier.

## Install or update the project

Open PowerShell in the folder containing `pyproject.toml`:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
```

If `pyproject.toml` is not shown by `Get-ChildItem`, you are in the wrong folder.

## Recommended two-scenario experiment

Download the 10 required highway and urban archives:

```powershell
python download_data.py --profile recommended --output data/raw
```

Preview the tuning plan:

```powershell
python optimize_experiments.py --plan-only
```

Run the complete recommended optimization:

```powershell
python optimize_experiments.py --profile recommended --raw-dir data/raw --processed-root data/optimized --runs-dir runs/optimization --seeds 42 43 44
```

This requires substantial time and disk space because sequence lengths 16, 24,
and 32 are stored separately. An NVIDIA Colab runtime is strongly recommended.
The AMD Radeon 860M does not provide CUDA to standard PyTorch, so local training
uses the CPU.

## Short CPU functionality check

This only checks the optimizer and must not be reported as final performance:

```powershell
python optimize_experiments.py --profile recommended --raw-dir data/raw --processed-root data/optimized --runs-dir runs/optimization_check --max-tuning-trials 2 --epochs 3 --seeds 42 --no-hierarchical
```

## Final full-dataset experiment

Download all four scenarios and all selected attacks:

```powershell
python download_data.py --profile full --output data/raw
```

Tune on the recommended two-scenario subset, then automatically retrain the
selected setup on all four scenarios:

```powershell
python optimize_experiments.py --profile recommended --final-full --raw-dir data/raw --processed-root data/optimized --runs-dir runs/full_optimization --seeds 42 43 44
```

Use this for the final research result if storage and GPU time allow.

## Output files

| File | Meaning |
|---|---|
| `tuning_results.csv` | Validation results for every hyperparameter trial |
| `multi_seed_validation.csv` | Validation results for the frozen configuration |
| `multi_seed_test.csv` | Final Test results for every seed |
| `optimization_summary.json` | Mean, standard deviation, minimum, and maximum metrics |
| `selected_config.yaml` | Exact winning configuration |
| `best_model.pt` | Deployable checkpoint selected by Validation performance |

Each seed directory also contains training curves, confusion matrices, and
per-class reports.

## How the hierarchical option works

The standard model directly predicts three classes. The optional hierarchy
first estimates `Normal versus Attack`. For the Attack branch it then estimates
`Sybil versus Illusion`. These probabilities are combined into valid Normal,
Sybil, and Illusion probabilities, so evaluation and inference use the same
interface as the original model.

The hierarchy is not selected merely because it ran. It must beat the flat
model using the same Validation selection rule.

## Safe restart behaviour

Completed compatible trials are reused. If a run directory contains a different
configuration, the program stops instead of silently overwriting it. Use a new
`--runs-dir` for a new experiment.

Only use these options when you intentionally want to replace the exact generated
optimization directories:

```powershell
python optimize_experiments.py ... --overwrite-runs --overwrite-processed
```

## What to report

- Exact processed sequence counts from each manifest
- Mean and standard deviation across final seeds
- Accuracy, Balanced Accuracy, Macro-F1, MCC, and false-positive rate
- Normal, Sybil, and Illusion precision, recall, and F1
- Final confusion matrices
- Whether the flat or hierarchical classifier was selected
- Whether class weighting or balanced sampling was selected

Do not choose a configuration after looking at Test results. Do not claim RSSI
was used unless genuine measurements were supplied during preprocessing and the
model was retrained.

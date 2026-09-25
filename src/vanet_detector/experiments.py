from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from .constants import ATTACK_TO_CLASS
from .download import PROFILE_SCENARIOS
from .evaluate import evaluate_checkpoint
from .preprocess import prepare_dataset
from .training import train
from .utils import load_config, write_json


def required_archive_names(profile: str) -> set[str]:
    if profile not in PROFILE_SCENARIOS:
        raise ValueError(f"Unknown profile {profile!r}; choose {sorted(PROFILE_SCENARIOS)}")
    names: set[str] = set()
    for scenario in PROFILE_SCENARIOS[profile]:
        names.add(f"InTAS_{scenario}.zip")
        for attack in ATTACK_TO_CLASS:
            names.add(f"InTAS_{scenario}_{attack}.zip")
    return names


def validate_raw_profile(raw_dir: Path, profile: str) -> None:
    required = required_archive_names(profile)
    available = {path.name for path in Path(raw_dir).glob("*.zip")}
    missing = sorted(required - available)
    if missing:
        preview = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"The {profile} experiment is missing {len(missing)} required archive(s):\n"
            f"  - {preview}\n"
            f"Run: python download_data.py --profile {profile} --output {raw_dir}"
        )


def curated_tuning_trials() -> list[dict[str, object]]:
    """Small, controlled search that changes one important choice at a time."""
    return [
        {
            "name": "baseline_lr1e3",
            "sequence_length": 16,
            "learning_rate": 1e-3,
            "dropout": 0.25,
            "sampling": "class_weight",
            "classification_mode": "flat",
        },
        {
            "name": "lr5e4",
            "sequence_length": 16,
            "learning_rate": 5e-4,
            "dropout": 0.25,
            "sampling": "class_weight",
            "classification_mode": "flat",
        },
        {
            "name": "lr3e4",
            "sequence_length": 16,
            "learning_rate": 3e-4,
            "dropout": 0.25,
            "sampling": "class_weight",
            "classification_mode": "flat",
        },
        {
            "name": "dropout20",
            "sequence_length": 16,
            "learning_rate": 5e-4,
            "dropout": 0.20,
            "sampling": "class_weight",
            "classification_mode": "flat",
        },
        {
            "name": "dropout30",
            "sequence_length": 16,
            "learning_rate": 5e-4,
            "dropout": 0.30,
            "sampling": "class_weight",
            "classification_mode": "flat",
        },
        {
            "name": "sequence24",
            "sequence_length": 24,
            "learning_rate": 5e-4,
            "dropout": 0.25,
            "sampling": "class_weight",
            "classification_mode": "flat",
        },
        {
            "name": "sequence32",
            "sequence_length": 32,
            "learning_rate": 5e-4,
            "dropout": 0.25,
            "sampling": "class_weight",
            "classification_mode": "flat",
        },
        {
            "name": "balanced_sampler",
            "sequence_length": 16,
            "learning_rate": 5e-4,
            "dropout": 0.25,
            "sampling": "balanced",
            "classification_mode": "flat",
        },
    ]


def build_trial_config(
    base_config: dict,
    trial: dict[str, object],
    data_dir: Path,
    seed: int,
    epochs: int | None = None,
) -> dict:
    config = copy.deepcopy(base_config)
    config["seed"] = int(seed)
    config["data_dir"] = str(data_dir)
    config.setdefault("model", {})["dropout"] = float(trial["dropout"])
    config["model"]["classification_mode"] = str(trial["classification_mode"])
    training = config.setdefault("training", {})
    training["learning_rate"] = float(trial["learning_rate"])
    training["class_weighting"] = trial["sampling"] == "class_weight"
    training["balanced_sampling"] = trial["sampling"] == "balanced"
    if epochs is not None:
        training["epochs"] = int(epochs)
    return config


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _processed_directory(root: Path, profile: str, sequence_length: int) -> Path:
    return root / f"{profile}_seq{sequence_length}"


def ensure_processed_dataset(
    raw_dir: Path,
    processed_root: Path,
    profile: str,
    sequence_length: int,
    stride: int,
    overwrite: bool,
) -> Path:
    output_dir = _processed_directory(processed_root, profile, sequence_length)
    expected_archives = required_archive_names(profile)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        compatible = (
            int(manifest.get("seq_len", -1)) == sequence_length
            and int(manifest.get("stride", -1)) == stride
            and set(manifest.get("archives", [])) == expected_archives
        )
        if compatible:
            return output_dir
        if not overwrite:
            raise ValueError(
                f"Existing processed data at {output_dir} does not match this experiment. "
                "Choose another --processed-root or pass --overwrite-processed."
            )

    prepare_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        seq_len=sequence_length,
        stride=stride,
        overwrite=overwrite,
        scenarios=list(PROFILE_SCENARIOS[profile]),
    )
    manifest = _read_json(manifest_path)
    if set(manifest.get("attack_types", [])) != set(ATTACK_TO_CLASS):
        raise RuntimeError("Processed data does not contain all required Sybil/Illusion attacks")
    for split in ("train", "validation", "test"):
        counts = manifest["splits"][split]["class_counts"]
        missing_classes = [name for name, count in counts.items() if int(count) == 0]
        if missing_classes:
            raise RuntimeError(f"{split} is missing classes: {missing_classes}")
    return output_dir


def _metric_row(trial: dict[str, object], metrics: dict, run_dir: Path) -> dict:
    return {
        **trial,
        "run_dir": str(run_dir),
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "accuracy": metrics["accuracy"],
        "false_positive_rate": metrics["false_positive_rate"],
        "normal_f1": metrics["per_class"]["normal"]["f1"],
        "sybil_f1": metrics["per_class"]["sybil"]["f1"],
        "illusion_f1": metrics["per_class"]["illusion"]["f1"],
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _selection_key(row: dict) -> tuple[float, float, float, float]:
    return (
        float(row["macro_f1"]),
        float(row["illusion_f1"]),
        float(row["balanced_accuracy"]),
        -float(row["false_positive_rate"]),
    )


def _run_validation_trial(
    config: dict,
    run_dir: Path,
    overwrite_runs: bool,
) -> tuple[Path, dict]:
    config["run_dir"] = str(run_dir)
    metrics_path = run_dir / "validation_metrics.json"
    checkpoint_path = run_dir / "best_model.pt"
    if metrics_path.exists() and checkpoint_path.exists() and not overwrite_runs:
        saved_config = _read_json(run_dir / "resolved_config.json")
        if saved_config != config:
            raise ValueError(
                f"{run_dir} contains a different experiment. Choose a new --runs-dir "
                "or pass --overwrite-runs."
            )
        return checkpoint_path, _read_json(metrics_path)
    checkpoint = train(
        config,
        run_dir,
        overwrite=overwrite_runs,
        evaluate_test=False,
    )
    return checkpoint, _read_json(metrics_path)


def _aggregate(rows: list[dict], keys: list[str]) -> dict:
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return summary


def run_optimization(
    base_config_path: Path,
    raw_dir: Path,
    processed_root: Path,
    runs_dir: Path,
    profile: str = "recommended",
    stride: int = 2,
    seeds: tuple[int, ...] = (42, 43, 44),
    epochs: int | None = None,
    max_tuning_trials: int = 0,
    hierarchical: bool = True,
    hierarchical_trigger_f1: float = 0.65,
    final_full: bool = False,
    overwrite_processed: bool = False,
    overwrite_runs: bool = False,
) -> Path:
    if profile not in {"recommended", "full"}:
        raise ValueError("Optimization profile must be 'recommended' or 'full'")
    if not seeds:
        raise ValueError("At least one seed is required")
    validate_raw_profile(raw_dir, "full" if final_full else profile)
    base_config = load_config(base_config_path)
    runs_dir.mkdir(parents=True, exist_ok=True)

    trials = curated_tuning_trials()
    if max_tuning_trials > 0:
        trials = trials[:max_tuning_trials]

    tuning_rows: list[dict] = []
    for index, trial in enumerate(trials, start=1):
        sequence_length = int(trial["sequence_length"])
        data_dir = ensure_processed_dataset(
            raw_dir,
            processed_root,
            profile,
            sequence_length,
            stride,
            overwrite_processed,
        )
        config = build_trial_config(base_config, trial, data_dir, seed=seeds[0], epochs=epochs)
        run_dir = runs_dir / "tuning" / f"{index:02d}_{trial['name']}"
        _, metrics = _run_validation_trial(config, run_dir, overwrite_runs)
        tuning_rows.append(_metric_row(trial, metrics, run_dir))
        _write_rows(runs_dir / "tuning_results.csv", tuning_rows)

    selected_row = max(tuning_rows, key=_selection_key)
    selected_trial = {
        key: selected_row[key]
        for key in (
            "name",
            "sequence_length",
            "learning_rate",
            "dropout",
            "sampling",
            "classification_mode",
        )
    }
    weakest_attack_f1 = min(float(selected_row["sybil_f1"]), float(selected_row["illusion_f1"]))
    if hierarchical and weakest_attack_f1 < hierarchical_trigger_f1:
        hierarchy_trial = selected_trial | {
            "name": f"hierarchical_from_{selected_trial['name']}",
            "classification_mode": "hierarchical",
        }
        data_dir = _processed_directory(
            processed_root, profile, int(hierarchy_trial["sequence_length"])
        )
        config = build_trial_config(
            base_config, hierarchy_trial, data_dir, seed=seeds[0], epochs=epochs
        )
        run_dir = runs_dir / "tuning" / f"{len(tuning_rows) + 1:02d}_{hierarchy_trial['name']}"
        _, metrics = _run_validation_trial(config, run_dir, overwrite_runs)
        hierarchy_row = _metric_row(hierarchy_trial, metrics, run_dir)
        tuning_rows.append(hierarchy_row)
        _write_rows(runs_dir / "tuning_results.csv", tuning_rows)
        selected_row = max(tuning_rows, key=_selection_key)
        selected_trial = {
            key: selected_row[key]
            for key in (
                "name",
                "sequence_length",
                "learning_rate",
                "dropout",
                "sampling",
                "classification_mode",
            )
        }

    final_profile = "full" if final_full else profile
    sequence_length = int(selected_trial["sequence_length"])
    final_data_dir = ensure_processed_dataset(
        raw_dir,
        processed_root,
        final_profile,
        sequence_length,
        stride,
        overwrite_processed,
    )

    stability_rows: list[dict] = []
    checkpoints: list[tuple[Path, dict, dict]] = []
    for seed in seeds:
        config = build_trial_config(
            base_config, selected_trial, final_data_dir, seed=seed, epochs=epochs
        )
        run_dir = runs_dir / "final_seeds" / f"seed_{seed}"
        checkpoint, validation_metrics = _run_validation_trial(
            config, run_dir, overwrite_runs
        )
        row = _metric_row(selected_trial | {"seed": seed}, validation_metrics, run_dir)
        stability_rows.append(row)
        checkpoints.append((checkpoint, config, row))
        _write_rows(runs_dir / "multi_seed_validation.csv", stability_rows)

    # Hyperparameters are now frozen. Test is evaluated once per final seed for reporting.
    test_rows: list[dict] = []
    for checkpoint, _, validation_row in checkpoints:
        run_dir = Path(str(validation_row["run_dir"]))
        test_metrics = evaluate_checkpoint(
            checkpoint_path=checkpoint,
            processed_dir=final_data_dir,
            split="test",
            output_dir=run_dir,
        )
        test_rows.append(
            _metric_row(
                selected_trial | {"seed": validation_row["seed"]}, test_metrics, run_dir
            )
        )
        _write_rows(runs_dir / "multi_seed_test.csv", test_rows)

    best_validation = max(stability_rows, key=_selection_key)
    best_seed = int(best_validation["seed"])
    best_checkpoint = next(item[0] for item in checkpoints if int(item[2]["seed"]) == best_seed)
    final_checkpoint = runs_dir / "best_model.pt"
    shutil.copy2(best_checkpoint, final_checkpoint)
    selected_config = next(item[1] for item in checkpoints if int(item[2]["seed"]) == best_seed)
    with (runs_dir / "selected_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(selected_config, handle, sort_keys=False)

    summary_keys = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "false_positive_rate",
        "normal_f1",
        "sybil_f1",
        "illusion_f1",
    ]
    summary = {
        "selection_rule": (
            "Highest validation Macro-F1; ties prefer Illusion F1, balanced accuracy, "
            "then lower false-positive rate"
        ),
        "tuning_profile": profile,
        "final_profile": final_profile,
        "selected_trial": selected_trial,
        "reported_seeds": list(seeds),
        "best_checkpoint_seed_by_validation": best_seed,
        "validation_summary": _aggregate(stability_rows, summary_keys),
        "test_summary": _aggregate(test_rows, summary_keys),
        "best_model": str(final_checkpoint),
    }
    write_json(runs_dir / "optimization_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return final_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune on Validation, optionally compare a hierarchical classifier, then "
            "report multiple frozen seeds on Test"
        )
    )
    parser.add_argument("--base-config", type=Path, default=Path("configs/recommended.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/optimized"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/optimization"))
    parser.add_argument("--profile", choices=("recommended", "full"), default="recommended")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, help="Override maximum epochs; omit for 50")
    parser.add_argument(
        "--max-tuning-trials",
        type=int,
        default=0,
        help="Limit the curated search for a shorter trial run; 0 runs all eight",
    )
    parser.add_argument(
        "--hierarchical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try hierarchy automatically when either attack-class F1 is below the trigger",
    )
    parser.add_argument("--hierarchical-trigger-f1", type=float, default=0.65)
    parser.add_argument(
        "--final-full",
        action="store_true",
        help="Tune on the chosen profile, then run the selected setup on all four scenarios",
    )
    parser.add_argument("--overwrite-processed", action="store_true")
    parser.add_argument("--overwrite-runs", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the curated trials without preprocessing or training",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.plan_only:
        print(json.dumps(curated_tuning_trials(), indent=2))
        return
    run_optimization(
        base_config_path=args.base_config,
        raw_dir=args.raw_dir,
        processed_root=args.processed_root,
        runs_dir=args.runs_dir,
        profile=args.profile,
        stride=args.stride,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        max_tuning_trials=args.max_tuning_trials,
        hierarchical=args.hierarchical,
        hierarchical_trigger_f1=args.hierarchical_trigger_f1,
        final_full=args.final_full,
        overwrite_processed=args.overwrite_processed,
        overwrite_runs=args.overwrite_runs,
    )


if __name__ == "__main__":
    main()

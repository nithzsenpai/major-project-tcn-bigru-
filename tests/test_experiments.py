from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml

from vanet_detector.experiments import (
    build_trial_config,
    curated_tuning_trials,
    required_archive_names,
    run_optimization,
    validate_raw_profile,
)


class ExperimentTests(unittest.TestCase):
    @staticmethod
    def _write_fixture_archive(path: Path, attacker: int) -> None:
        message = {
            "sender_id": "vehicle_1",
            "sender_alias": "alias_1",
            "messageID": "1",
            "sendTime": 1_000_000_000,
            "rcvTime": 1_001_000_000,
            "sender": {"pos": [10.0, 0.0, 0.0], "spd": 10.0, "acl": 0.0, "hed": 0.0},
            "receiver": {"pos": [0.0, 0.0, 0.0]},
            "attacker": attacker,
        }
        with zipfile.ZipFile(path, "w") as outer:
            for split in ("Train", "Validation", "Test"):
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as inner:
                    inner.writestr(
                        f"{split}/receiver_1.json", json.dumps([message])
                    )
                outer.writestr(f"{split}/{split}.zip", buffer.getvalue())

    def test_recommended_profile_includes_every_required_attack(self) -> None:
        names = required_archive_names("recommended")
        self.assertEqual(len(names), 10)
        for scenario in ("highway_2", "urban_2"):
            self.assertIn(f"InTAS_{scenario}.zip", names)
            self.assertIn(f"InTAS_{scenario}_trafficCongestionSybil.zip", names)
            self.assertIn(f"InTAS_{scenario}_constantPositionOffset.zip", names)
            self.assertIn(f"InTAS_{scenario}_randomPositionOffset.zip", names)
            self.assertIn(f"InTAS_{scenario}_positionMirroring.zip", names)

    def test_curated_search_covers_requested_optimization_axes(self) -> None:
        trials = curated_tuning_trials()
        self.assertEqual({trial["sequence_length"] for trial in trials}, {16, 24, 32})
        self.assertEqual(
            {trial["learning_rate"] for trial in trials}, {1e-3, 5e-4, 3e-4}
        )
        self.assertEqual({trial["dropout"] for trial in trials}, {0.20, 0.25, 0.30})
        self.assertEqual(
            {trial["sampling"] for trial in trials}, {"class_weight", "balanced"}
        )

    def test_balanced_trial_disables_class_weighting(self) -> None:
        base = {"model": {}, "training": {}}
        trial = next(
            item for item in curated_tuning_trials() if item["sampling"] == "balanced"
        )
        config = build_trial_config(base, trial, Path("processed"), seed=43)
        self.assertFalse(config["training"]["class_weighting"])
        self.assertTrue(config["training"]["balanced_sampling"])
        self.assertEqual(config["seed"], 43)

    def test_raw_profile_validation_reports_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "missing 10 required"):
                validate_raw_profile(Path(directory), "recommended")

    def test_tiny_optimizer_runs_validation_selection_before_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            for name in required_archive_names("recommended"):
                attacker = 0 if name in {
                    "InTAS_highway_2.zip",
                    "InTAS_urban_2.zip",
                } else 1
                self._write_fixture_archive(raw_dir / name, attacker)

            base_config = {
                "seed": 42,
                "deterministic": True,
                "data_dir": "unused",
                "model": {
                    "classification_mode": "flat",
                    "tcn_channels": [4],
                    "kernel_size": 3,
                    "gru_hidden": 2,
                    "relation_hidden": 2,
                    "bidirectional": True,
                    "attention_heads": 1,
                    "fusion_hidden": 8,
                    "dropout": 0.1,
                },
                "training": {
                    "batch_size": 2,
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "minimum_lr": 0.000001,
                    "weight_decay": 0.0001,
                    "focal_gamma": 1.0,
                    "label_smoothing": 0.0,
                    "class_weighting": True,
                    "balanced_sampling": False,
                    "auxiliary_weight": 0.1,
                    "max_gradient_norm": 1.0,
                    "early_stopping_patience": 1,
                    "lr_patience": 1,
                    "lr_factor": 0.5,
                    "num_workers": 0,
                    "amp": False,
                },
                "threshold": {"maximum_false_positive_rate": 0.5},
            }
            config_path = root / "base.yaml"
            config_path.write_text(yaml.safe_dump(base_config), encoding="utf-8")
            runs_dir = root / "runs"
            checkpoint = run_optimization(
                base_config_path=config_path,
                raw_dir=raw_dir,
                processed_root=root / "processed",
                runs_dir=runs_dir,
                seeds=(42,),
                epochs=1,
                max_tuning_trials=1,
                hierarchical=False,
            )
            self.assertTrue(checkpoint.exists())
            self.assertTrue((runs_dir / "optimization_summary.json").exists())
            self.assertTrue((runs_dir / "multi_seed_test.csv").exists())
            self.assertFalse(
                (runs_dir / "tuning" / "01_baseline_lr1e3" / "test_metrics.json").exists()
            )


if __name__ == "__main__":
    unittest.main()

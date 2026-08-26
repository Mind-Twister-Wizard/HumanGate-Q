from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from humangateq.agents import assess_workflows
from humangateq.data import DatasetError, load_dataset
from humangateq.metrics import evaluate_policies
from humangateq.modeling import fit_reliability_model
from humangateq.policies import (
    ACTIONS,
    ablation_actions,
    all_policy_actions,
    oracle_actions,
    tune_humangateq_thresholds,
)
from humangateq.scenarios import build_scenarios


def make_fixture(rows: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    qubits = rng.integers(2, 18, rows)
    depth = rng.integers(5, 180, rows)
    two = rng.integers(0, 60, rows)
    three = rng.integers(0, 12, rows)
    measurement = rng.integers(1, 20, rows)
    single = rng.integers(2, 120, rows)
    total = single + two + three + measurement
    latent = 1.0 - (0.0027 * depth + 0.0060 * two + 0.013 * three + 0.003 * qubits)
    latent += rng.normal(0, 0.08, rows)
    label = np.where(latent > 0.62, "HIGH", np.where(latent > 0.32, "MEDIUM", "LOW"))
    estimated_fidelity = np.clip(latent, 0, 1)
    frame = pd.DataFrame(
        {
            "circuit_name": [f"fixture_{index}" for index in range(rows)],
            "source_file": [f"fixture_{index}.qasm" for index in range(rows)],
            "number_of_qubits": qubits,
            "number_of_classical_bits": qubits,
            "depth": depth,
            "width": qubits * 2,
            "total_operations": total,
            "single_qubit_gates": single,
            "two_qubit_gates": two,
            "three_qubit_gates": three,
            "measurement_gates": measurement,
            "parameterized_gates": rng.integers(0, 20, rows),
            "entangling_gates": two + three,
            "parallelism_ratio": np.clip(total / np.maximum(depth * qubits, 1), 0, 1),
            "entangling_ratio": (two + three) / np.maximum(total, 1),
            "cx_depth": np.minimum(depth, two + 2 * three),
            "three_qubit_ratio": three / np.maximum(total, 1),
            "qubit_utilization": rng.uniform(0.2, 1.0, rows),
            "measurement_ratio": measurement / np.maximum(total, 1),
            "gate_diversity": rng.integers(2, 12, rows),
            "gate_x": rng.integers(0, 25, rows),
            "gate_h": rng.integers(0, 25, rows),
            "gate_cx": two,
            "estimated_fidelity": estimated_fidelity,
            "total_variation_distance": 1 - estimated_fidelity,
            "hellinger_distance": np.sqrt(1 - estimated_fidelity),
            "success_probability_ideal": np.ones(rows),
            "success_probability_noisy": estimated_fidelity,
            "reliability_class": label,
            "reliability_score": estimated_fidelity * 100,
        }
    )
    return frame


def risk_config() -> dict:
    return {
        "weights": {
            "reliability": 0.35,
            "predictive_uncertainty": 0.17,
            "distribution_shift": 0.13,
            "verification": 0.16,
            "resource_complexity": 0.06,
            "workflow_criticality": 0.13,
        },
        "interactions": {
            "reliability_x_criticality": 0.18,
            "verification_x_criticality": 0.14,
        },
        "thresholds": {
            "repair": 0.30,
            "human_review": 0.50,
            "abstain": 0.76,
            "high_stakes_review": 0.38,
            "high_stakes_criticality": 0.85,
            "direct_low_probability_abstain": 0.72,
        },
    }


class HumanGateQTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temporary.name) / "training_dataset.csv"
        make_fixture().to_csv(self.dataset_path, index=False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_loader_removes_target_leakage(self) -> None:
        bundle = load_dataset(self.dataset_path, maximum_rows=700, random_seed=42)
        forbidden = {
            "estimated_fidelity",
            "reliability_score",
            "total_variation_distance",
            "success_probability_noisy",
        }
        self.assertTrue(forbidden.isdisjoint(bundle.features.columns))
        self.assertEqual(set(bundle.target.unique()), {"HIGH", "MEDIUM", "LOW"})

    def test_invalid_schema_is_rejected(self) -> None:
        invalid_path = Path(self.temporary.name) / "invalid.csv"
        pd.DataFrame({"a": [1, 2], "reliability_class": ["HIGH", "LOW"]}).to_csv(
            invalid_path, index=False
        )
        with self.assertRaises(DatasetError):
            load_dataset(invalid_path)

    def test_end_to_end_policy_components(self) -> None:
        data = load_dataset(self.dataset_path, maximum_rows=800, random_seed=42)
        model = fit_reliability_model(
            data.features,
            data.target,
            {
                "n_estimators": 25,
                "max_depth": 12,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "n_jobs": 1,
            },
            test_fraction=0.25,
            calibration_fraction=0.10,
            policy_validation_fraction=0.10,
            random_seed=42,
            calibration_bins=6,
        )
        scenario_config = {
            "maximum_test_cases": 180,
            "proportions": {
                "clean": 0.50,
                "ambiguous_goal": 0.15,
                "tool_failure": 0.10,
                "metadata_missing": 0.10,
                "distribution_shift": 0.15,
            },
        }
        scenarios = build_scenarios(
            model.bundle,
            model.x_test,
            model.y_test,
            data.identifiers,
            scenario_config,
            42,
        )
        config = risk_config()
        assessment = assess_workflows(model.bundle, scenarios.features, scenarios.metadata, config)
        oracle = oracle_actions(assessment)
        actions = all_policy_actions(assessment, config["thresholds"])
        metrics = evaluate_policies(actions, oracle)
        ablated, risks = ablation_actions(assessment, config)

        self.assertEqual(set(actions["HumanGate-Q"].unique()).difference(ACTIONS), set())
        self.assertTrue(assessment["risk_score"].between(0, 1).all())
        self.assertIn("HumanGate-Q", metrics.index)
        self.assertEqual(ablated.shape, risks.shape)
        self.assertEqual(
            list(ablated.columns),
            [
                "Full HumanGate-Q",
                "Without reliability",
                "Without predictive uncertainty",
                "Without structural shift",
                "Without explicit shift flag",
                "Without both shift pathways",
                "Without verification",
                "Without resource risk",
                "Without criticality",
                "Without interactions",
            ],
        )
        self.assertTrue(
            ablated["Full HumanGate-Q"].equals(actions["HumanGate-Q"])
        )
        self.assertTrue(np.isfinite(model.metrics["expected_calibration_error"]))
        self.assertGreater(len(model.x_policy_validation), 0)

        tuned, search = tune_humangateq_thresholds(
            assessment,
            oracle,
            config["thresholds"],
            {
                "enabled": True,
                "target_unsafe_execution_rate": 0.50,
                "minimum_escalation_recall": 0.20,
                "minimum_safe_automation_coverage": 0.20,
                "maximum_human_review_rate": 0.80,
                "grid": {
                    "repair": [0.25, 0.30],
                    "human_review": [0.48, 0.50],
                    "abstain": [0.74, 0.76],
                    "high_stakes_review": [0.38],
                    "direct_low_probability_abstain": [0.72],
                },
            },
        )
        self.assertEqual(int(search["selected"].sum()), 1)
        self.assertLessEqual(tuned["repair"], tuned["human_review"])
        self.assertLessEqual(tuned["human_review"], tuned["abstain"])


if __name__ == "__main__":
    unittest.main()

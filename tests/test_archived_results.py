from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper_run"


class ArchivedResultTests(unittest.TestCase):
    def test_headline_metrics_match_chapter(self) -> None:
        model = pd.read_csv(RESULTS / "tables" / "model_metrics.csv").set_index(
            "metric"
        )["value"]
        policy = pd.read_csv(
            RESULTS / "tables" / "policy_metrics.csv", index_col=0
        ).loc["HumanGate-Q"]

        self.assertAlmostEqual(float(model["accuracy"]), 0.8955, places=12)
        self.assertAlmostEqual(float(model["macro_f1"]), 0.8753886861094754)
        self.assertAlmostEqual(
            float(model["expected_calibration_error"]), 0.007449496990961081
        )
        self.assertAlmostEqual(float(policy["exact_action_accuracy"]), 0.7440)
        self.assertAlmostEqual(
            float(policy["unsafe_execution_rate"]), 0.09694793536804308
        )
        self.assertAlmostEqual(
            float(policy["appropriate_escalation_recall"]), 0.876296117675428
        )
        self.assertAlmostEqual(float(policy["safe_automation_coverage"]), 0.9003516998827668)
        self.assertAlmostEqual(float(policy["human_review_rate"]), 0.1434)

    def test_archived_configuration_and_actions(self) -> None:
        metadata = json.loads((RESULTS / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["selected_candidate"], "LGB-S15-C40")
        self.assertEqual(metadata["dataset"]["rows"], 30000)
        self.assertEqual(metadata["dataset"]["features"], 28)
        self.assertEqual(
            metadata["dataset"]["sha256"],
            "506eff548af0603f771290cb75d3515f203d976162f3171b6494a1bb4d94025c",
        )
        decisions = pd.read_csv(RESULTS / "tables" / "workflow_decisions.csv")
        self.assertEqual(
            decisions["action__HumanGate-Q"].value_counts().to_dict(),
            {"ABSTAIN": 3002, "SELF_REPAIR": 724, "ASK_HUMAN": 717, "EXECUTE": 557},
        )

    def test_expanded_ablation_and_figure_masters(self) -> None:
        ablation = pd.read_csv(RESULTS / "tables" / "ablation_metrics.csv", index_col=0)
        self.assertEqual(len(ablation), 10)
        self.assertEqual(
            list(ablation.index),
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

        expected_hashes = {
            "14-01.tif": "e7583ae63c8b1c58f6aa7d29d1b7d3087e2615813caf8e8e84aa16606e2fa5d0",
            "14-02.tif": "8492fe551a83546966b342bcefc1b0b9d75552a2e9fca1124df7f6b043b3733e",
            "14-03.tif": "cf74cd0b41df4239e74de4131950143aaa7cb651a6dabf18bc40dbeee078a274",
            "14-04.tif": "be27b095517747f57da205001b580e00ae2898775b9d09cd6f0060fce208724d",
            "14-05.tif": "42f05c9a7bb1a532a856b48a9532a236d8ae834e1f2b5db2842048c622b3f271",
            "14-06.tif": "636ca6e6477d27827bc024b7d18ac561b5f8b1c3687801ee4e72b2afce0277d4",
            "14-07.tif": "2a0ebd1404a30606460a4a4fe80446c61d343f795c55d6aadafdba27e00931c3",
            "14-08.tif": "454e9eabe2b456d148ca09a57939a1314165dddc995cf39ac25ad95a98bca0d5",
            "14-09.tif": "da9727b5ff1ef7129feeeb5e65cfea1d3f992efc2c81511854715e0ede1e02a1",
            "14-10.tif": "5e854b775b3f3a6a4a82b92e8f400c15ca42abbeaa8f372fb96c1533006859b3",
        }
        figure_root = RESULTS / "publication_figures_tif"
        for filename, expected in expected_hashes.items():
            digest = hashlib.sha256((figure_root / filename).read_bytes()).hexdigest()
            self.assertEqual(digest, expected)

    def test_raw_dataset_is_not_redistributed(self) -> None:
        files = sorted(path.name for path in (ROOT / "data" / "raw").iterdir())
        self.assertEqual(files, ["README.md"])


if __name__ == "__main__":
    unittest.main()

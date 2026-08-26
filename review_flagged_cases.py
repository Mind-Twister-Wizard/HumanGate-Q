"""Optional console interface for real human review of HumanGate-Q referrals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CHOICES = {
    "a": "APPROVE_EXECUTION",
    "r": "REQUEST_REPAIR",
    "b": "BLOCK_EXECUTION",
    "s": "SKIP",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", default="outputs/latest/tables/workflow_decisions.csv")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--log", default="outputs/latest/tables/human_review_log.csv")
    args = parser.parse_args()

    decisions_path = Path(args.decisions)
    if not decisions_path.exists():
        print("No completed decision table was found. Run RUN_EXPERIMENT.bat first.")
        return 2
    decisions = pd.read_csv(decisions_path)
    action_column = "action__HumanGate-Q"
    if action_column not in decisions:
        print(f"Required column is missing from {decisions_path}: {action_column}")
        return 2
    queue = decisions[decisions[action_column] == "ASK_HUMAN"].head(max(0, args.limit))
    if queue.empty:
        print("HumanGate-Q did not refer any cases for human review in this run.")
        return 0

    print("HumanGate-Q Expert Review Console")
    print("Choices: [A]pprove, request [R]epair, [B]lock, [S]kip, [Q]uit")
    records = []
    for _, row in queue.iterrows():
        print("\n" + "=" * 68)
        print(f"Circuit:       {row.get('circuit_id', 'unknown')}")
        print(f"Domain:        {row.get('workflow_domain', 'unknown')}")
        print(f"Scenario:      {row.get('scenario', 'unknown')}")
        print(f"Predicted:     {row.get('predicted_reliability', 'unknown')}")
        print(f"Risk score:    {float(row.get('risk_score', 0.0)):.3f}")
        print(f"Criticality:   {float(row.get('workflow_criticality', 0.0)):.3f}")
        print(f"Primary cause: {row.get('dominant_risk_driver', 'unknown')}")
        while True:
            choice = input("Decision [A/R/B/S/Q]: ").strip().lower()
            if choice == "q":
                queue = queue.iloc[0:0]
                break
            if choice in CHOICES:
                records.append(
                    {
                        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "circuit_id": row.get("circuit_id", "unknown"),
                        "risk_score": row.get("risk_score", 0.0),
                        "agent_action": row.get(action_column),
                        "human_decision": CHOICES[choice],
                    }
                )
                break
            print("Enter A, R, B, S, or Q.")
        if choice == "q":
            break

    if records:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        new_records = pd.DataFrame(records)
        if log_path.exists():
            previous = pd.read_csv(log_path)
            new_records = pd.concat([previous, new_records], ignore_index=True)
        new_records.to_csv(log_path, index=False)
        print(f"\nSaved {len(records)} review decision(s) to {log_path}")
    else:
        print("\nNo decisions were recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


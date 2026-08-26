# Quick Start on Windows

## Recommended route

1. Extract the repository ZIP.
2. Open the inner `HumanGate-Q` folder.
3. Double-click **`START_HERE.bat`**.

Keep the internet connected during the first run. The script creates `.venv`,
installs the package, downloads QUASAR from Kaggle, verifies the source and
archived artifacts, then runs the full experiment.

## If the automatic Kaggle download fails

1. Open the [QUASAR dataset page](https://www.kaggle.com/datasets/ykmadhav/synthetic-quantum-circuit-reliability-dataset).
2. Sign in if Kaggle requests it, download the archive, and extract its files
   into `HumanGate-Q\data\raw\`.
3. Double-click **`RUN_EXPERIMENT.bat`**.

Do not rename the CSV columns. The loader automatically selects the CSV that
contains `reliability_class`.

## Useful launchers

- `RUN_EXPERIMENT.bat` — repeat the complete experiment.
- `RUN_QUICK_TEST.bat` — run a smaller environment check.
- `VERIFY_PACKAGE.bat` — verify dependencies, tests, archived evidence, and a
  downloaded dataset when present.
- `REVIEW_FLAGGED_CASES.bat` — inspect workflows assigned `ASK_HUMAN`.

Fresh results appear under `outputs\latest\`; the immutable chapter evidence is
under `results\paper_run\`.

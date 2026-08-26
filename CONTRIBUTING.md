# Contributing

Thank you for helping improve HumanGate-Q. Please open an issue before a large
change and describe the scientific or engineering motivation.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python verify_package.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Pull-request checklist

- Keep the Kaggle dataset and credentials out of commits.
- Preserve train/calibration/policy-validation/test separation.
- Add or update tests when policy behavior changes.
- Run `python verify_package.py` and `ruff check .`.
- Describe whether results differ from `results/paper_run/`.

Changing an operating threshold or oracle rule creates a new experimental
condition and must be reported as such; it must not silently replace the
archived chapter run.

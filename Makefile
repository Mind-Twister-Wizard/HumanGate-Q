.PHONY: setup download verify quick run clean-outputs

setup:
	python -m pip install -r requirements.txt

download:
	python download_dataset.py

verify:
	python verify_package.py

quick:
	python run_all.py --quick --download-if-missing

run:
	python run_all.py --download-if-missing

clean-outputs:
	@echo "Remove outputs/ manually if you want a clean rerun; archived results/paper_run is immutable."

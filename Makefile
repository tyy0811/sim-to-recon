.PHONY: deploy data build baseline stress figures test lint clean all

# ---- Modal deployment ----
# One-time setup: deploys sfm_dtu_scan9 and dense_mvs_subset to Modal.
deploy:
	modal deploy modal_app.py

# ---- Data ----
# DTU scan9 is downloaded to a Modal volume (not local disk) since the
# reconstruction pipeline runs on Modal A10G. Requires `make deploy` first.
data:
	@echo "Downloading DTU scan9 to Modal volume (simtorecon-dtu-data)..."
	modal run modal_app.py::download_dtu_scan9

# ---- C++ build ----
build:
	cmake -B build cpp/calib
	cmake --build build

# ---- Reconstruction ----
# Calls deployed Modal functions. Run `make deploy && make data` once first.
baseline:
	python experiments/run_baseline.py --n-views 49

stress:
	python experiments/run_stress_view_count.py --seeds 42,123,7

# ---- Figures ----
figures:
	python experiments/generate_figures.py

# ---- Quality ----
test:
	pytest tests/ -v --cov=simtorecon
	@if [ -d "build" ]; then cd build && ctest --output-on-failure; fi

lint:
	ruff check src/ tests/ experiments/

# ---- Housekeeping ----
clean:
	rm -rf build/
	rm -rf results/
	rm -rf docs/figures/*.png

all: lint test

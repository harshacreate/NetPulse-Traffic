# NetPulse-Traffic — Telecommunications Traffic Forecasting

**Sector:** Telecommunications  
**Task:** Hourly network throughput regression

Forecasts synthetic network throughput from daily and weekly patterns, temperature, rain, event bursts, and past utilization. A maintenance flag can produce temporary drops.

## Data and split

`netpulse/data.py` contains a deterministic offline simulator. It uses no public or private operational data and does not download a dataset. Values and patterns are illustrative. Reported target units are **relative synthetic throughput units**; they are generated for this simulator and are not calibrated real-world measurements. The holdout is chronological (first 75% train, last 25% test). Evaluation uses a rolling one-step forecast: actual observed history up to the row is available to the lag features. This is not recursive multi-step evaluation.

## Models and metrics

- **Baseline:** 24-hour seasonal naive (`lag_24`).
- **Stronger model:** histogram gradient-boosting regression fitted on the training period.
- **Metrics:** MAE, RMSE, R², sMAPE, 90th-percentile absolute error, and mean bias. Error summaries are emitted by the CLI; no performance is asserted in advance.

## Limitations

The simulator is not tied to a carrier, topology, or measurement unit and does not model routing, customer mix, outages, or network upgrades. No operational capacity decisions should rely on this demo.

## Run

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ".[test]"
    python -m netpulse.train --seed 42 --samples 1200 --json
    python -m pytest -q

`make setup`, `make train`, and `make test` are shortcuts. Python 3.9 or newer is required.

## Layout

- `netpulse/data.py`: domain simulator and lag features.
- `netpulse/split.py`: forward-only split.
- `netpulse/model.py`: seasonal baseline, stronger model, metrics.
- `netpulse/train.py`: CLI.
- `tests/`: reproducibility, split, metrics, and CLI tests.

No code license has been selected; default copyright rules apply. Choose a license before public distribution.

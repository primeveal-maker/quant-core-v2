# Quant Core v2

Quant Core v2 is a dependency-light research scaffold that ingests synthetic order-book events, derives regime-aware features, calibrates a lightweight logistic model, and evaluates portfolio decisions with execution-aware backtesting.

## Project layout

```
project/
├── configs/          # YAML-like configuration files
├── data/             # Storage for raw/clean/features (created at runtime)
├── logs/             # Rotating application logs
├── reports/          # Metrics, equity curves, trade logs (JSON)
├── signals/          # Prediction artefacts
├── src/              # Source modules
└── tests/            # Pytest suite
```

## Key modules

* `src/ingest.py` — normalize L2 events, annotate latency, and persist as JSON-lines parquet stubs.
* `src/feature_amk.py` — adaptive microstructure kernel features computed from event windows.
* `src/flow_field.py` — simplified flow-field divergence trigger.
* `src/feature_entropy.py` — Shannon entropy, KL regime detection, and sizing signals.
* `src/topology_graph.py` — mutual-information topology graph with centrality indicators.
* `src/features.py` — end-to-end feature builder combining AMK, flow, entropy, and OHLC bars.
* `src/train.py` — walk-forward splitter, quantile/robust scalers, and logistic training loop.
* `src/meta_fusion.py` — agreement-density meta fusion with entropy gating.
* `src/risk.py` — entropy-aware position sizing and rudimentary VaR proxy.
* `src/exec_engine.py` — impact/slippage aware execution simulator.
* `src/backtest.py` — lightweight backtester producing trades, equity, and metrics JSON artefacts.

## CLI usage

```
python -m src.ingest --exchange binance --symbols ETHUSDT,BTCUSDT --live false --since 2023-01-01
python -m src.features --tf 30m --symbols ETHUSDT --configs configs/features.yaml --exchange binance
python -m src.train --config configs/train.yaml --symbols ETHUSDT --tf 30m --exchange binance
python -m src.backtest --config configs/backtest.yaml --features data/features/binance/ETHUSDT/features_ETHUSDT_30m.json
python -m src.live --config configs/live.yaml --paper true
```

## Testing

```
pytest
```

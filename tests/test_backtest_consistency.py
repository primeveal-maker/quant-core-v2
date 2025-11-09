from __future__ import annotations

from datetime import datetime, timedelta

from src.backtest import Backtester, BacktestConfig


def build_feature_row(ts: datetime, price: float) -> dict:
    return {
        "ts": ts.isoformat(),
        "open": price,
        "close": price + 1.0,
        "spread": 0.1,
        "flow_trigger": 1.0,
        "regime_flag": "trend",
        "H_dir": 0.2,
    }


def test_execution_delay_and_fees():
    config = BacktestConfig(
        execution={"mode": "next_bar_open", "delay_bars": 1, "fees_bps": 10.0, "impact_coef": 0.0, "slippage_model": "impact"},
        risk={"base_size": 1.0, "entropy_sizing": False, "k_sigma_stop": 2.0, "max_symbol_risk": 1.0, "var_alpha": 0.05},
        meta_fusion={"agree_low": 0.0, "agree_high": 0.0},
    )
    backtester = Backtester(config)
    start = datetime(2023, 1, 1)
    features = [build_feature_row(start + timedelta(minutes=i), 100 + i) for i in range(5)]
    outputs = backtester.run(features)
    trades = outputs["trades"]
    assert trades, "expected at least one trade"
    first_trade = trades[0]
    assert first_trade["px_req"] == 101.0
    assert first_trade["fees"] > 0.0

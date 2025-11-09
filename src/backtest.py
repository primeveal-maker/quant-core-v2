from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .exec_engine import ExecutionConfig, ExecutionEngine
from .meta_fusion import MetaFusionConfig, MetaSignalFusion
from .risk import RiskConfig, RiskManager
from .utils.config import load_config
from .utils.logging import configure_logging, get_logger


logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    execution: Dict[str, float]
    risk: Dict[str, float]
    meta_fusion: Dict[str, float]


class Backtester:
    def __init__(self, config: BacktestConfig) -> None:
        self.exec_engine = ExecutionEngine(ExecutionConfig(**config.execution))
        self.risk_manager = RiskManager(RiskConfig(**config.risk))
        self.meta_fusion = MetaSignalFusion(MetaFusionConfig(**config.meta_fusion))

    def run(self, features: List[Dict[str, float | str]]) -> Dict[str, List[Dict[str, float | str]]]:
        signals_input: List[Dict[str, float]] = []
        market: List[Dict[str, float | str]] = []
        for row in features:
            signals_input.append(
                {
                    "ml_signal": float(row.get("y_pred", row.get("flow_trigger", 0.0))),
                    "flow_signal": float(row.get("flow_trigger", 0.0)),
                    "entropy_signal": 1.0 if row.get("regime_flag") == "trend" else -0.5,
                    "topology_bias": float(row.get("topology_weight", 0.0)),
                    "entropy": float(abs(row.get("H_dir", 1.0)) + 1e-3),
                    "H_entropy": float(abs(row.get("H_dir", 1.0)) + 1e-3),
                }
            )
            market.append(
                {
                    "ts": row.get("ts"),
                    "open": float(row.get("open", row.get("close", 0.0))),
                    "close": float(row.get("close", 0.0)),
                    "spread": float(row.get("spread", 0.0)),
                }
            )
        fused = self.meta_fusion.combine(signals_input)
        for entry in fused:
            entry["alpha"] = abs(entry.get("meta_signal", 0.0))
            entry["H_entropy"] = entry.get("H_entropy", 1.0)
            entry["size"] = self.risk_manager.position_size(entry)
        reports = self.exec_engine.execute(fused, market)
        trades = self.exec_engine.to_dicts()

        pnl: List[float] = []
        equity = 0.0
        equity_curve: List[Dict[str, float | str]] = []
        for report in reports:
            side = 1 if report.side == "BUY" else -1
            market_idx = next((i for i, bar in enumerate(market) if bar["ts"] == report.ts), None)
            if market_idx is None or market_idx + 1 >= len(market):
                continue
            exit_price = float(market[market_idx + 1]["close"])
            pnl_trade = side * report.qty * (exit_price - report.px_fill) - report.fees - report.slip
            equity += pnl_trade
            pnl.append(pnl_trade)
            equity_curve.append({"ts": report.ts, "equity": equity, "drawdown": min(0.0, pnl_trade)})
        metrics = compute_metrics(pnl, equity_curve)
        return {"trades": trades, "equity": equity_curve, "metrics": [metrics]}


def compute_metrics(pnl: List[float], equity_curve: List[Dict[str, float | str]]) -> Dict[str, float]:
    if not pnl:
        return {"sharpe": 0.0, "sortino": 0.0, "profit_factor": 0.0, "win_rate": 0.0, "max_dd": 0.0}
    mean = sum(pnl) / len(pnl)
    variance = sum((x - mean) ** 2 for x in pnl) / len(pnl)
    std = variance ** 0.5
    sharpe = mean / std if std else 0.0
    downside = [x for x in pnl if x < 0]
    if downside:
        down_var = sum(x ** 2 for x in downside) / len(downside)
        sortino = mean / (down_var ** 0.5) if down_var else 0.0
    else:
        sortino = 0.0
    wins = sum(1 for x in pnl if x > 0)
    losses = sum(-x for x in pnl if x < 0)
    profit = sum(x for x in pnl if x > 0)
    profit_factor = profit / losses if losses else 0.0
    win_rate = wins / len(pnl)
    max_dd = min((entry.get("drawdown", 0.0) for entry in equity_curve), default=0.0)
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "max_dd": max_dd,
    }


def backtest_cli(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quant Core backtest")
    parser.add_argument("--config", default="configs/backtest.yaml")
    parser.add_argument("--features", required=True)

    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_logging(Path("logs"))
    cfg = load_config(Path(args.config)).data

    feature_path = Path(args.features)
    rows = [json.loads(line) for line in feature_path.read_text(encoding="utf-8").splitlines()]
    backtester = Backtester(BacktestConfig(**cfg))
    outputs = backtester.run(rows)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    prefix = feature_path.stem
    (reports_dir / f"{prefix}_trades.json").write_text("\n".join(json.dumps(row) for row in outputs["trades"]), encoding="utf-8")
    (reports_dir / f"{prefix}_equity_curve.json").write_text("\n".join(json.dumps(row) for row in outputs["equity"]), encoding="utf-8")
    (reports_dir / f"{prefix}_metrics.json").write_text("\n".join(json.dumps(row) for row in outputs["metrics"]), encoding="utf-8")
    logger.info("Backtest completed for %s", feature_path)


if __name__ == "__main__":  # pragma: no cover
    backtest_cli()

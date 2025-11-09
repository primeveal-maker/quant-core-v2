from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ExecutionConfig:
    mode: str = "next_bar_open"
    delay_bars: int = 1
    fees_bps: float = 6.0
    impact_coef: float = 0.6
    slippage_model: str = "impact"


@dataclass
class ExecutionReport:
    order_id: int
    ts: str
    side: str
    qty: float
    px_req: float
    px_fill: float
    slip: float
    fees: float


class ExecutionEngine:
    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config
        self._order_id = 0
        self.reports: List[ExecutionReport] = []

    def execute(self, signals: List[dict], market: List[dict]) -> List[ExecutionReport]:
        for idx, row in enumerate(signals):
            self._order_id += 1
            future_idx = idx + self.config.delay_bars
            if future_idx >= len(market):
                continue
            bar = market[future_idx]
            side = "BUY" if row.get("meta_signal", 0.0) > 0 else "SELL"
            qty = abs(row.get("size", 0.0))
            requested = bar.get("open", bar.get("close", 0.0))
            spread = bar.get("spread", 0.0)
            if self.config.slippage_model == "impact":
                slip = self.config.impact_coef * qty * spread
            else:
                slip = spread * 0.5
            fill = requested + slip if side == "BUY" else requested - slip
            fees = abs(fill * qty) * self.config.fees_bps / 10_000
            self.reports.append(
                ExecutionReport(
                    order_id=self._order_id,
                    ts=str(bar.get("ts")),
                    side=side,
                    qty=qty,
                    px_req=requested,
                    px_fill=fill,
                    slip=slip,
                    fees=fees,
                )
            )
        return self.reports

    def to_dicts(self) -> List[dict]:
        return [report.__dict__ for report in self.reports]

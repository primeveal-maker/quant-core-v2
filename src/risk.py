from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class RiskConfig:
    base_size: float = 1.0
    entropy_sizing: bool = True
    k_sigma_stop: float = 2.5
    max_symbol_risk: float = 0.07
    var_alpha: float = 0.05


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def position_size(self, row: dict) -> float:
        entropy = abs(row.get("H_entropy", 1.0)) if self.config.entropy_sizing else 1.0
        entropy_factor = 1.0 / max(entropy, 1e-6)
        size = self.config.base_size * row.get("alpha", 1.0) * row.get("confidence", 1.0) * entropy_factor
        return max(-self.config.max_symbol_risk, min(self.config.max_symbol_risk, size))

    def stop_loss(self, volatility: float) -> float:
        return self.config.k_sigma_stop * volatility

    def var_limit(self, pnl: List[float]) -> float:
        if not pnl:
            return 0.0
        sorted_pnl = sorted(pnl)
        index = int(len(sorted_pnl) * self.config.var_alpha)
        index = min(max(index, 0), len(sorted_pnl) - 1)
        return sorted_pnl[index]

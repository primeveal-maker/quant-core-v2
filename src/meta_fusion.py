from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MetaFusionConfig:
    agree_low: float = 0.45
    agree_high: float = 0.65


class MetaSignalFusion:
    def __init__(self, config: MetaFusionConfig) -> None:
        self.config = config

    def combine(self, rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []
        for row in rows:
            ml = row.get("ml_signal", 0.0)
            flow = row.get("flow_signal", 0.0)
            entropy_signal = row.get("entropy_signal", 0.0)
            topo = row.get("topology_bias", 0.0)
            entropy = max(row.get("entropy", 1.0), 1e-6)
            votes = [1 if ml > 0 else -1 if ml < 0 else 0, 1 if flow > 0 else -1 if flow < 0 else 0, 1 if entropy_signal > 0 else -1 if entropy_signal < 0 else 0, 1 if topo > 0 else -1 if topo < 0 else 0]
            weights = [0.4, 0.2, 0.2, 0.2]
            score = sum(v * w for v, w in zip(votes, weights))
            consensus = 1 if score > 0 else -1 if score < 0 else 0
            if consensus == 0:
                agreement = 0.0
            else:
                matches = sum(1 for v in votes if v == consensus)
                agreement = matches / len(votes)
            entropy_sizing = 1.0 / entropy
            if agreement < self.config.agree_low or entropy_sizing <= 0.5:
                meta = 0.0
            elif agreement > self.config.agree_high:
                meta = consensus * agreement
            else:
                meta = consensus * 0.5
            confidence = max(min(agreement * entropy_sizing, 1.0), 0.0)
            enriched = dict(row)
            enriched["meta_signal"] = meta
            enriched["confidence"] = confidence
            enriched["agreement"] = agreement
            results.append(enriched)
        return results

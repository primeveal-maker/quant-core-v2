from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List

from .data_types import L2Event


@dataclass
class AMKFeature:
    ts: datetime
    imbalance_g: float
    spread_grad: float
    cancel_ratio: float
    aggr_markets_ratio: float
    latency_proxy: float
    weighted_spread: float

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "ts": self.ts.isoformat(),
            "imbalance_g": self.imbalance_g,
            "spread_grad": self.spread_grad,
            "cancel_ratio": self.cancel_ratio,
            "aggr_markets_ratio": self.aggr_markets_ratio,
            "latency_proxy": self.latency_proxy,
            "weighted_spread": self.weighted_spread,
        }


class AdaptiveMicrostructureKernel:
    def __init__(self, window_sec: int) -> None:
        self.window_sec = window_sec

    def _window(self, events: List[L2Event], end_index: int) -> List[L2Event]:
        end_ts = events[end_index].exch_ts
        start_ts = end_ts - timedelta(seconds=self.window_sec)
        result: List[L2Event] = []
        for idx in range(end_index, -1, -1):
            event = events[idx]
            if event.exch_ts < start_ts:
                break
            result.append(event)
        return list(reversed(result))

    def _compute_feature(self, window_events: List[L2Event]) -> AMKFeature:
        spreads = [e.ask_px - e.bid_px for e in window_events]
        spread_grad = 0.0
        if len(spreads) >= 2:
            diffs = [spreads[i] - spreads[i - 1] for i in range(1, len(spreads))]
            spread_grad = sum(diffs) / len(diffs)
        total_ask = sum(e.ask_sz for e in window_events)
        total_bid = sum(e.bid_sz for e in window_events)
        denom = total_ask + total_bid if (total_ask + total_bid) else 1.0
        imbalance_g = (total_ask - total_bid) / denom
        latency_proxy = sum((e.recv_ts - e.exch_ts).total_seconds() for e in window_events) / max(len(window_events), 1)
        weighted_spread = sum(spreads) / max(len(spreads), 1)
        return AMKFeature(
            ts=window_events[-1].exch_ts,
            imbalance_g=imbalance_g,
            spread_grad=spread_grad,
            cancel_ratio=0.0,
            aggr_markets_ratio=0.0,
            latency_proxy=latency_proxy,
            weighted_spread=weighted_spread,
        )

    def transform(self, events: Iterable[L2Event]) -> List[AMKFeature]:
        ordered = sorted(events, key=lambda e: e.exch_ts)
        features: List[AMKFeature] = []
        for idx in range(len(ordered)):
            window_events = self._window(ordered, idx)
            if not window_events:
                continue
            features.append(self._compute_feature(window_events))
        return features


def build_amk_features(events: Iterable[L2Event], config: Dict[str, List[int]]) -> List[Dict[str, float | str]]:
    result: List[Dict[str, float | str]] = []
    windows = config.get("window_sec", [10])
    event_list = list(events)
    for window in windows:
        kernel = AdaptiveMicrostructureKernel(window)
        for feature in kernel.transform(event_list):
            feature_dict = feature.to_dict()
            feature_dict = {f"amk_{window}_{k}": v for k, v in feature_dict.items() if k != "ts"}
            feature_dict["ts"] = feature.ts.isoformat()
            result.append(feature_dict)
    result.sort(key=lambda item: item["ts"])
    return result

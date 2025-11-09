from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List

from .data_types import L2Event


@dataclass
class FlowFieldPoint:
    ts: datetime
    divergence: float
    trigger: float

    def to_dict(self) -> Dict[str, float | str]:
        return {"ts": self.ts.isoformat(), "divF": self.divergence, "flow_trigger": self.trigger}


class FlowFieldCalculator:
    def compute(self, events: Iterable[L2Event]) -> List[FlowFieldPoint]:
        ordered = sorted(events, key=lambda e: e.exch_ts)
        points: List[FlowFieldPoint] = []
        prev_depth = None
        prev_flow = None
        for event in ordered:
            depth = event.ask_sz - event.bid_sz
            flow = event.ask_px - event.bid_px
            if prev_depth is None:
                divergence = 0.0
            else:
                divergence = (depth - prev_depth) + (flow - prev_flow)
            trigger = max(min(divergence, 1.0), -1.0)
            points.append(FlowFieldPoint(ts=event.exch_ts, divergence=divergence, trigger=trigger))
            prev_depth = depth
            prev_flow = flow
        return points


def build_flow_field(events: Iterable[L2Event]) -> List[Dict[str, float | str]]:
    calculator = FlowFieldCalculator()
    return [point.to_dict() for point in calculator.compute(events)]

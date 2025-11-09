from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


@dataclass
class L2Event:
    exch_ts: datetime
    recv_ts: datetime
    bid_px: float
    ask_px: float
    bid_sz: float
    ask_sz: float
    event_id: int | None = None
    seq: int | None = None

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "exch_ts": self.exch_ts.isoformat(),
            "recv_ts": self.recv_ts.isoformat(),
            "bid_px": self.bid_px,
            "ask_px": self.ask_px,
            "bid_sz": self.bid_sz,
            "ask_sz": self.ask_sz,
            "event_id": self.event_id,
            "seq": self.seq,
        }


@dataclass
class Bar:
    ts_open: datetime
    ts_close: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "ts_open": self.ts_open.isoformat(),
            "ts_close": self.ts_close.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "spread": self.spread,
        }


def sort_events(events: List[L2Event]) -> List[L2Event]:
    return sorted(events, key=lambda e: e.exch_ts)

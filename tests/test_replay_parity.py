from __future__ import annotations

from datetime import datetime, timedelta

from src.data_types import L2Event
from src.feature_amk import build_amk_features


def test_amk_parity_between_sorted_and_unsorted():
    events = []
    base = datetime(2023, 1, 1)
    for i in range(20):
        events.append(
            L2Event(
                exch_ts=base + timedelta(seconds=i),
                recv_ts=base + timedelta(seconds=i, milliseconds=5),
                bid_px=100 + i * 0.01,
                ask_px=100.1 + i * 0.01,
                bid_sz=1 + i * 0.1,
                ask_sz=1.5 + i * 0.1,
            )
        )
    sorted_features = build_amk_features(events, {"window_sec": [5]})
    shuffled = list(reversed(events))
    unsorted_features = build_amk_features(shuffled, {"window_sec": [5]})
    assert sorted_features == unsorted_features

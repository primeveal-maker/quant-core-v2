from __future__ import annotations

from datetime import datetime, timedelta

from src.train import WalkForwardSplitter


def test_walk_forward_monotonicity():
    timestamps = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(120)]
    splitter = WalkForwardSplitter({"train_months": 2, "valid_months": 1, "test_months": 1})
    splits = splitter.split(timestamps)
    assert splits
    for train_idx, valid_idx, test_idx in splits:
        assert train_idx and valid_idx and test_idx
        assert max(train_idx) < min(valid_idx)
        assert max(valid_idx) < min(test_idx)

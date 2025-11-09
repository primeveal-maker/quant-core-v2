from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_unix_millis(dt: datetime) -> int:
    return int(ensure_utc(dt).timestamp() * 1000)


def moving_window(values: Iterable[float], window: int) -> List[List[float]]:
    values_list = list(values)
    if window <= 0:
        raise ValueError("window must be positive")
    result: List[List[float]] = []
    for idx in range(len(values_list)):
        start = max(0, idx - window + 1)
        result.append(values_list[start : idx + 1])
    return result

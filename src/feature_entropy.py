from __future__ import annotations

from typing import Dict, Iterable, List

from .data_types import Bar


def _histogram(values: List[float], bins: int) -> List[int]:
    if not values:
        return [0] * bins
    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        hist = [0] * bins
        hist[0] = len(values)
        return hist
    step = (max_v - min_v) / bins or 1.0
    hist = [0] * bins
    for value in values:
        idx = int((value - min_v) / step)
        if idx >= bins:
            idx = bins - 1
        hist[idx] += 1
    return hist


def shannon_entropy(values: List[float]) -> float:
    bins = min(len(values), 10) or 1
    hist = _histogram(values, bins)
    total = sum(hist)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in hist:
        if count == 0:
            continue
        p = count / total
        entropy -= p * (0 if p <= 0 else _log2(p))
    return entropy


def _log2(value: float) -> float:
    import math

    return math.log(value, 2)


def compute_entropy_features(bars: Iterable[Bar], window: int, thresholds: Dict[str, float]) -> List[Dict[str, float | str]]:
    bar_list = list(bars)
    results: List[Dict[str, float | str]] = []
    price_changes: List[float] = []
    returns: List[float] = []
    volumes: List[float] = []
    prev_close = None
    for bar in bar_list:
        if prev_close is None:
            change = 0.0
            ret = 0.0
        else:
            change = bar.close - prev_close
            ret = (bar.close - prev_close) / prev_close if prev_close else 0.0
        prev_close = bar.close
        price_changes.append(change)
        returns.append(ret)
        volumes.append(bar.volume)
        window_changes = price_changes[-window:]
        window_returns = returns[-window:]
        window_volumes = volumes[-window:]
        h_dir = shannon_entropy([1 if c > 0 else -1 if c < 0 else 0 for c in window_changes])
        h_ret = shannon_entropy(window_returns)
        h_vol = shannon_entropy(window_volumes)
        prev_entropy = results[-1] if results else {"H_dir": 1.0, "H_ret": 1.0, "H_vol": 1.0}
        p = [abs(prev_entropy["H_dir"]), abs(prev_entropy["H_ret"]), abs(prev_entropy["H_vol"])]
        q = [abs(h_dir), abs(h_ret), abs(h_vol)]
        p_total = sum(p) or 1.0
        q_total = sum(q) or 1.0
        p = [value / p_total for value in p]
        q = [value / q_total for value in q]
        kl = 0.0
        for q_i, p_i in zip(q, p):
            if q_i == 0 or p_i == 0:
                continue
            kl += q_i * (q_i / p_i)
        if kl > thresholds.get("D_KL", 0.5):
            regime = "transition"
        elif h_dir < thresholds.get("H_low", 0.4):
            regime = "trend"
        elif h_dir > thresholds.get("H_high", 0.7):
            regime = "choppy"
        else:
            regime = "neutral"
        results.append(
            {
                "ts": bar.ts_close.isoformat(),
                "H_dir": h_dir,
                "H_ret": h_ret,
                "H_vol": h_vol,
                "D_KL": kl,
                "regime_flag": regime,
            }
        )
    return results

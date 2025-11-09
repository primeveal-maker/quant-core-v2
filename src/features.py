from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

from .data_types import Bar, L2Event
from .feature_amk import build_amk_features
from .feature_entropy import compute_entropy_features
from .flow_field import build_flow_field
from .utils.config import load_config
from .utils.logging import configure_logging, get_logger


logger = get_logger(__name__)


def _parse_event(record: Dict[str, object]) -> L2Event:
    return L2Event(
        exch_ts=datetime.fromisoformat(str(record["exch_ts"])),
        recv_ts=datetime.fromisoformat(str(record["recv_ts"])),
        bid_px=float(record["bid_px"]),
        ask_px=float(record["ask_px"]),
        bid_sz=float(record["bid_sz"]),
        ask_sz=float(record["ask_sz"]),
        event_id=int(record.get("event_id", 0)),
        seq=int(record.get("seq", 0)),
    )


def load_clean_events(base_path: Path, exchange: str, symbol: str) -> List[L2Event]:
    events: List[L2Event] = []
    symbol_path = base_path / "clean" / exchange / symbol
    if not symbol_path.exists():
        raise FileNotFoundError(f"No normalized data for {symbol}")
    for file in symbol_path.rglob("*.parquet"):
        for line in file.read_text(encoding="utf-8").splitlines():
            events.append(_parse_event(json.loads(line)))
    events.sort(key=lambda e: e.exch_ts)
    return events


def build_bars(events: List[L2Event], interval: timedelta) -> List[Bar]:
    if not events:
        return []
    bars: List[Bar] = []
    bucket_start = events[0].exch_ts
    bucket_end = bucket_start + interval
    open_px = events[0].bid_px
    high_px = open_px
    low_px = open_px
    close_px = open_px
    volume = 0.0
    for event in events:
        while event.exch_ts >= bucket_end:
            spread = high_px - low_px
            bars.append(Bar(bucket_start, bucket_end, open_px, high_px, low_px, close_px, volume, spread))
            bucket_start = bucket_end
            bucket_end = bucket_start + interval
            open_px = event.bid_px
            high_px = open_px
            low_px = open_px
            close_px = open_px
            volume = 0.0
        price = event.bid_px
        high_px = max(high_px, price)
        low_px = min(low_px, price)
        close_px = price
        volume += event.bid_sz + event.ask_sz
    spread = high_px - low_px
    bars.append(Bar(bucket_start, bucket_end, open_px, high_px, low_px, close_px, volume, spread))
    return bars


def merge_feature_sets(base: Dict[str, Dict[str, float | str]], features: List[Dict[str, float | str]]) -> None:
    for row in features:
        ts = str(row.get("ts"))
        if not ts:
            continue
        base.setdefault(ts, {}).update({k: v for k, v in row.items() if k != "ts"})


def build_features(events: List[L2Event], cfg: Dict[str, Dict]) -> List[Dict[str, float | str]]:
    combined: Dict[str, Dict[str, float | str]] = {}
    amk_features = build_amk_features(events, cfg.get("amk", {}))
    flow_features = build_flow_field(events)
    bars = build_bars(events, timedelta(minutes=1))
    entropy_cfg = cfg.get("entropy", {})
    entropy_features = compute_entropy_features(
        bars,
        entropy_cfg.get("window_bars", 100),
        entropy_cfg.get("thresholds", {}),
    )

    merge_feature_sets(combined, amk_features)
    merge_feature_sets(combined, flow_features)
    merge_feature_sets(
        combined,
        [
            {
                "ts": bar.ts_close.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "spread": bar.spread,
            }
            for bar in bars
        ],
    )
    merge_feature_sets(combined, entropy_features)

    rows = []
    for ts in sorted(combined.keys()):
        row = {"ts": ts}
        row.update(combined[ts])
        rows.append(row)
    return rows


def feature_cli(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Feature builder")
    parser.add_argument("--tf", default="1m")
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--configs", default="configs/features.yaml")
    parser.add_argument("--exchange", default="binance")

    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_logging(Path("logs"))
    cfg = load_config(Path(args.configs)).data
    base_path = Path(cfg.get("storage", {}).get("base_path", "data"))
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    for symbol in symbols:
        events = load_clean_events(base_path, args.exchange, symbol)
        rows = build_features(events, cfg)
        out_dir = base_path / "features" / args.exchange / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"features_{symbol}_{args.tf}.json"
        out_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        logger.info("Saved %d feature rows to %s", len(rows), out_path)


if __name__ == "__main__":  # pragma: no cover
    feature_cli()

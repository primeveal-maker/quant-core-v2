from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from .data_types import L2Event, sort_events
from .utils.config import load_config
from .utils.hash import hash_paths
from .utils.logging import configure_logging, get_logger


logger = get_logger(__name__)


@dataclass
class IngestConfig:
    exchange: str
    symbols: List[str]
    live: bool
    since: Optional[datetime]


class IngestNormalizer:
    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.clean_path = base_path / "clean"
        self.clean_path.mkdir(parents=True, exist_ok=True)

    def normalize_l2(self, events: List[L2Event], exchange: str, symbol: str, date: datetime) -> Path:
        ordered = sort_events(events)
        output_records = []
        for idx, event in enumerate(ordered):
            event.event_id = idx
            event.seq = idx
            clock_skew_ms = (event.recv_ts - event.exch_ts).total_seconds() * 1000
            latency_ms = abs(clock_skew_ms)
            record = event.to_dict()
            record["clock_skew_ms"] = clock_skew_ms
            record["latency_ms"] = latency_ms
            output_records.append(record)

        out_dir = self.clean_path / exchange / symbol / date.strftime("%Y-%m-%d")
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / f"l2_{symbol}_{date:%Y-%m-%d}.parquet"
        file_path.write_text("\n".join(json.dumps(r) for r in output_records), encoding="utf-8")
        logger.info("Saved %d normalized events to %s", len(output_records), file_path)
        return file_path


def ingest_cli(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Quant Core v2 ingest")
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--live", default="false")
    parser.add_argument("--since")
    parser.add_argument("--config", default="configs/data.yaml")

    args = parser.parse_args(list(argv) if argv is not None else None)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    live = str(args.live).lower() == "true"
    since = datetime.fromisoformat(args.since) if args.since else None

    configure_logging(Path("logs"))
    cfg_bundle = load_config(Path(args.config))
    logger.info("Loaded config %s hash=%s", args.config, cfg_bundle.hash)
    _ = IngestNormalizer(Path(cfg_bundle.data.get("storage", {}).get("base_path", "data")))
    logger.info(
        "Ingest stub invoked exchange=%s symbols=%s live=%s since=%s",
        args.exchange,
        symbols,
        live,
        since,
    )


def compute_code_hash() -> str:
    src_paths = [Path("src"), Path("configs")]
    files: List[Path] = []
    for base in src_paths:
        if base.exists():
            for path in base.rglob("*.py"):
                files.append(path)
            for path in base.rglob("*.yaml"):
                files.append(path)
    return hash_paths(files)


if __name__ == "__main__":  # pragma: no cover
    ingest_cli()

from __future__ import annotations

import argparse
from pathlib import Path

from .utils.config import load_config
from .utils.logging import configure_logging, get_logger


logger = get_logger(__name__)


def live_cli(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Quant Core live simulator")
    parser.add_argument("--config", default="configs/live.yaml")
    parser.add_argument("--paper", default="true")

    args = parser.parse_args(argv)
    configure_logging(Path("logs"))
    cfg = load_config(Path(args.config)).data if Path(args.config).exists() else {}
    logger.info("Starting live simulator with config %s (paper=%s)", cfg, args.paper)


if __name__ == "__main__":  # pragma: no cover
    live_cli()

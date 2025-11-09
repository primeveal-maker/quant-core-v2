from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .utils.config import load_config
from .utils.logging import configure_logging, get_logger


logger = get_logger(__name__)


@dataclass
class SplitConfig:
    train_months: int
    valid_months: int
    test_months: int


class WalkForwardSplitter:
    def __init__(self, config: Dict[str, int]) -> None:
        self.config = SplitConfig(
            train_months=config.get("train_months", 12),
            valid_months=config.get("valid_months", 3),
            test_months=config.get("test_months", 3),
        )

    def split(self, timestamps: Sequence[datetime]) -> List[Tuple[List[int], List[int], List[int]]]:
        if not timestamps:
            return []
        splits: List[Tuple[List[int], List[int], List[int]]] = []
        start = min(timestamps)
        end = max(timestamps)
        cursor = start
        while cursor < end:
            train_end = cursor + _months(self.config.train_months)
            valid_end = train_end + _months(self.config.valid_months)
            test_end = valid_end + _months(self.config.test_months)
            train_idx = [i for i, ts in enumerate(timestamps) if cursor <= ts < train_end]
            valid_idx = [i for i, ts in enumerate(timestamps) if train_end <= ts < valid_end]
            test_idx = [i for i, ts in enumerate(timestamps) if valid_end <= ts < test_end]
            if not test_idx:
                break
            splits.append((train_idx, valid_idx, test_idx))
            cursor = cursor + _months(self.config.test_months)
        return splits


def _months(count: int) -> datetime:
    # approximate month as 30 days for simplicity
    from datetime import timedelta

    return timedelta(days=30 * count)


class QuantileScaler:
    def __init__(self) -> None:
        self.mins: List[float] = []
        self.maxs: List[float] = []

    def fit(self, data: List[List[float]]) -> None:
        if not data:
            return
        columns = len(data[0])
        self.mins = [min(row[i] for row in data) for i in range(columns)]
        self.maxs = [max(row[i] for row in data) for i in range(columns)]

    def transform(self, row: List[float]) -> List[float]:
        if not self.mins:
            return row
        transformed = []
        for value, min_v, max_v in zip(row, self.mins, self.maxs):
            if max_v == min_v:
                transformed.append(0.0)
            else:
                transformed.append((value - min_v) / (max_v - min_v))
        return transformed


class RobustScaler:
    def __init__(self) -> None:
        self.medians: List[float] = []
        self.iqrs: List[float] = []

    def fit(self, data: List[List[float]]) -> None:
        if not data:
            return
        columns = len(data[0])
        sorted_cols = [[row[i] for row in data] for i in range(columns)]
        for col in sorted_cols:
            col.sort()
        self.medians = [col[len(col) // 2] for col in sorted_cols]
        self.iqrs = []
        for col in sorted_cols:
            q1 = col[len(col) // 4]
            q3 = col[(len(col) * 3) // 4]
            self.iqrs.append(max(q3 - q1, 1e-6))

    def transform(self, row: List[float]) -> List[float]:
        if not self.medians:
            return row
        return [(value - median) / iqr for value, median, iqr in zip(row, self.medians, self.iqrs)]


def select_scaler(name: str):
    if name == "quantile":
        return QuantileScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError(f"Unknown scaler {name}")


class LogisticModel:
    def __init__(self, learning_rate: float = 0.05, epochs: int = 50) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: List[float] = []
        self.bias: float = 0.0

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        if not X:
            return
        features = len(X[0])
        self.weights = [0.0] * features
        self.bias = 0.0
        for _ in range(self.epochs):
            for row, target in zip(X, y):
                z = self.bias + sum(w * value for w, value in zip(self.weights, row))
                pred = _sigmoid(z)
                error = pred - target
                self.bias -= self.learning_rate * error
                for i in range(features):
                    self.weights[i] -= self.learning_rate * error * row[i]

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        return [_sigmoid(self.bias + sum(w * value for w, value in zip(self.weights, row))) for row in X]

    def predict(self, X: List[List[float]]) -> List[int]:
        return [1 if prob >= 0.5 else 0 for prob in self.predict_proba(X)]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def load_feature_rows(paths: List[Path]) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for path in paths:
        if path.suffix == ".json":
            for line in path.read_text(encoding="utf-8").splitlines():
                rows.append(json.loads(line))
        elif path.suffix == ".csv":
            # very small parser for comma separated values with header
            with path.open("r", encoding="utf-8") as fh:
                header = fh.readline().strip().split(",")
                for line in fh:
                    values = line.strip().split(",")
                    rows.append({key: value for key, value in zip(header, values)})
    return rows


def extract_features(rows: List[Dict[str, float | str]]) -> Tuple[List[datetime], List[List[float]], List[int]]:
    timestamps: List[datetime] = []
    features: List[List[float]] = []
    target: List[int] = []
    closes: List[float] = []
    for row in rows:
        ts = datetime.fromisoformat(str(row["ts"]))
        timestamps.append(ts)
        closes.append(float(row.get("close", 0.0)))
        numeric_values = []
        for key, value in row.items():
            if key in {"ts", "symbol", "regime_flag"}:
                continue
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
        features.append(numeric_values)
    for i in range(len(closes) - 1):
        ret = 1 if closes[i + 1] > closes[i] else 0
        target.append(ret)
    target.append(0)
    return timestamps, features, target


def train_cli(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Quant Core model")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--tf", default="1m")
    parser.add_argument("--exchange", default="binance")

    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_logging(Path("logs"))
    cfg = load_config(Path(args.config)).data

    base_path = Path(cfg.get("features_path", "data/features"))
    feature_paths = [base_path / args.exchange / symbol.strip() / f"features_{symbol.strip()}_{args.tf}.json" for symbol in args.symbols.split(",")]
    rows = load_feature_rows(feature_paths)
    timestamps, features, target = extract_features(rows)

    splitter = WalkForwardSplitter(cfg.get("split", {}))
    splits = splitter.split(timestamps)
    if not splits:
        raise ValueError("Not enough data for walk-forward splitting")

    scaler = select_scaler(cfg.get("preprocess", {}).get("scaler", "quantile"))
    metrics_summary: List[Dict[str, float | str]] = []
    predictions_output: List[Dict[str, float | str]] = []

    for train_idx, valid_idx, test_idx in splits:
        train_X = [features[i] for i in train_idx]
        train_y = [target[i] for i in train_idx]
        valid_X = [features[i] for i in valid_idx]
        valid_y = [target[i] for i in valid_idx]
        test_X = [features[i] for i in test_idx]
        test_y = [target[i] for i in test_idx]

        scaler.fit(train_X)
        scaled_train = [scaler.transform(row) for row in train_X]
        scaled_valid = [scaler.transform(row) for row in valid_X]
        scaled_test = [scaler.transform(row) for row in test_X]

        model = LogisticModel()
        model.fit(scaled_train, train_y)
        preds = model.predict(scaled_test)
        proba = model.predict_proba(scaled_test)
        metrics_summary.append(
            {
                "split_start": timestamps[test_idx[0]].isoformat(),
                "accuracy": accuracy(test_y, preds),
            }
        )
        for idx, p in zip(test_idx, proba):
            predictions_output.append({"ts": timestamps[idx].isoformat(), "y_true": target[idx], "y_pred": p})

    signals_dir = Path("signals")
    reports_dir = Path("reports")
    signals_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = signals_dir / "predictions.json"
    metrics_path = reports_dir / "training_metrics.json"
    predictions_path.write_text("\n".join(json.dumps(row) for row in predictions_output), encoding="utf-8")
    metrics_path.write_text("\n".join(json.dumps(row) for row in metrics_summary), encoding="utf-8")
    logger.info("Saved predictions to %s", predictions_path)
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":  # pragma: no cover
    train_cli()

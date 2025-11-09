from __future__ import annotations

from src.train import QuantileScaler


def test_scaler_changes_with_additional_data():
    scaler = QuantileScaler()
    data_train = [[0.0], [1.0], [2.0]]
    scaler.fit(data_train)
    transformed_train = [scaler.transform(row) for row in data_train]
    scaler.fit(data_train + [[10.0]])
    transformed_expanded = [scaler.transform(row) for row in data_train]
    assert transformed_train != transformed_expanded

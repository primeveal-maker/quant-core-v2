from __future__ import annotations

from src.topology_graph import TopologyGraph, build_topology, topology_indicators


def test_topology_graph_weights():
    feature_frames = {
        "BTC": [{"ts": "2023-01-01T00:00:00", "close": float(i)} for i in range(10)],
        "ETH": [{"ts": "2023-01-01T00:00:00", "close": float(i + 1)} for i in range(10)],
    }
    graph, edges = build_topology(feature_frames, {"window_bars": 5, "graph_min_weight": 0.0})
    assert edges
    indicators = topology_indicators(graph)
    symbols = {row["symbol"] for row in indicators}
    assert symbols == {"BTC", "ETH"}
    for edge in edges:
        assert edge["weight"] >= 0.0

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class GraphEdge:
    a: str
    b: str
    weight: float


class TopologyGraph:
    def __init__(self) -> None:
        self.adj: Dict[str, Dict[str, float]] = {}

    def add_edge(self, a: str, b: str, weight: float) -> None:
        self.adj.setdefault(a, {})[b] = weight
        self.adj.setdefault(b, {})[a] = weight

    def degree(self, node: str) -> int:
        return len(self.adj.get(node, {}))

    def neighbors(self, node: str) -> List[str]:
        return list(self.adj.get(node, {}).keys())

    def nodes(self) -> List[str]:
        return list(self.adj.keys())

    def edges(self) -> List[GraphEdge]:
        seen = set()
        edges = []
        for a, neighbors in self.adj.items():
            for b, weight in neighbors.items():
                if (b, a) in seen:
                    continue
                seen.add((a, b))
                edges.append(GraphEdge(a, b, weight))
        return edges


def _hist(values: List[float], bins: int) -> List[int]:
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


def mutual_information(x: List[float], y: List[float], bins: int = 10) -> float:
    if not x or not y:
        return 0.0
    hist_x = _hist(x, bins)
    hist_y = _hist(y, bins)
    total_x = sum(hist_x) or 1
    total_y = sum(hist_y) or 1
    p_x = [count / total_x for count in hist_x]
    p_y = [count / total_y for count in hist_y]
    mi = 0.0
    for px, py in zip(p_x, p_y):
        if px == 0 or py == 0:
            continue
        mi += px * py
    return mi


def build_topology(feature_frames: Dict[str, List[Dict[str, float]]], config: Dict[str, float]) -> Tuple[TopologyGraph, List[Dict[str, float | str]]]:
    window = int(config.get("window_bars", 720))
    min_weight = float(config.get("graph_min_weight", 0.02))
    graph = TopologyGraph()
    edge_rows: List[Dict[str, float | str]] = []
    for a, b in itertools.combinations(sorted(feature_frames.keys()), 2):
        series_a = [row["close"] for row in feature_frames[a][-window:]]
        series_b = [row["close"] for row in feature_frames[b][-window:]]
        length = min(len(series_a), len(series_b))
        if length == 0:
            continue
        mi = mutual_information(series_a[-length:], series_b[-length:])
        if mi >= min_weight:
            graph.add_edge(a, b, mi)
            edge_rows.append({"symbol_a": a, "symbol_b": b, "weight": mi})
    return graph, edge_rows


def topology_indicators(graph: TopologyGraph) -> List[Dict[str, float | str]]:
    nodes = graph.nodes()
    indicators: List[Dict[str, float | str]] = []
    for node in nodes:
        neighbors = graph.neighbors(node)
        degree = len(neighbors)
        if degree <= 1:
            clustering = 0.0
        else:
            links = 0
            possible = degree * (degree - 1) / 2
            for a, b in itertools.combinations(neighbors, 2):
                if b in graph.adj.get(a, {}):
                    links += 1
            clustering = links / possible if possible else 0.0
        centrality = degree / max(len(nodes) - 1, 1)
        indicators.append({"symbol": node, "degree": degree, "centrality": centrality, "clustering": clustering})
    return indicators

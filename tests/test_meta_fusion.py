from __future__ import annotations

from src.meta_fusion import MetaFusionConfig, MetaSignalFusion


def test_low_agreement_disables_signal():
    fusion = MetaSignalFusion(MetaFusionConfig(agree_low=0.6, agree_high=0.8))
    rows = [
        {
            "ml_signal": 1.0,
            "flow_signal": -1.0,
            "entropy_signal": -1.0,
            "topology_bias": -1.0,
            "entropy": 2.0,
        },
        {
            "ml_signal": 1.0,
            "flow_signal": 1.0,
            "entropy_signal": 1.0,
            "topology_bias": 1.0,
            "entropy": 0.2,
        },
    ]
    fused = fusion.combine(rows)
    assert fused[0]["meta_signal"] == 0.0
    assert fused[1]["meta_signal"] > 0.0
    assert fused[0]["confidence"] < fused[1]["confidence"]

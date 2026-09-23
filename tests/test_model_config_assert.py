"""The config -> model architecture invariant.

A config field that is validated but never forwarded builds a model that silently
differs from the config.  That is not hypothetical here: a ``top_k: 1`` run once
trained a k=2 model and only surfaced as a shape error at the end of epoch 1.
"""
from types import SimpleNamespace

import pytest
import torch

from src.model import (
    DenseMoE16Adapter,
    PassthroughMoELayer,
    SpatialMoELayer,
    assert_model_matches_config,
)

DIM = 64


def _config(**overrides):
    fields = dict(num_experts=4, top_k=2, gate_mode="renormalized",
                  moe_type="sparse", moe_16_mode="sparse")
    fields.update(overrides)
    return SimpleNamespace(model=SimpleNamespace(**fields))


def _model(**overrides):
    """Build just the three MoE attributes the checker reads."""
    fields = dict(num_experts=4, k=2, gate_mode="renormalized", moe_type="sparse",
                  moe_16_mode="sparse")
    fields.update(overrides)
    layer_cls = {"sparse": SpatialMoELayer, "dense": DenseMoE16Adapter,
                 "none": PassthroughMoELayer}[fields["moe_type"]]
    kwargs = dict(dim=DIM)
    if layer_cls is SpatialMoELayer:
        kwargs.update(num_experts=fields["num_experts"], k=fields["k"],
                      gate_mode=fields["gate_mode"])
    moe = object.__new__(torch.nn.Module)  # bare container, no forward needed
    torch.nn.Module.__init__(moe)
    for name in ("moe_4", "moe_8"):
        setattr(moe, name, layer_cls(**kwargs))
    setattr(moe, "moe_16",
            DenseMoE16Adapter(dim=DIM) if fields["moe_16_mode"] == "dense"
            else layer_cls(**kwargs))
    return moe


def test_matching_config_passes():
    assert_model_matches_config(_model(), _config())


def test_top_k_mismatch_raises():
    # The historical bug: config says 1, the model was built with 2.
    with pytest.raises(ValueError, match=r"moe_4\.k"):
        assert_model_matches_config(_model(k=2), _config(top_k=1))


def test_expert_count_mismatch_raises():
    with pytest.raises(ValueError, match="num_experts"):
        assert_model_matches_config(_model(num_experts=8), _config(num_experts=4))


def test_gate_mode_mismatch_raises():
    with pytest.raises(ValueError, match="gate_mode"):
        assert_model_matches_config(_model(gate_mode="dense"), _config())


def test_moe_type_mismatch_raises():
    with pytest.raises(ValueError, match="moe_4"):
        assert_model_matches_config(_model(moe_type="dense"), _config())


def test_moe16_mode_mismatch_raises():
    with pytest.raises(ValueError, match="moe_16"):
        assert_model_matches_config(_model(moe_16_mode="dense"), _config())


def test_dense_arm_is_accepted():
    assert_model_matches_config(_model(moe_type="dense"), _config(moe_type="dense"))


def test_none_arm_is_accepted():
    assert_model_matches_config(_model(moe_type="none"), _config(moe_type="none"))

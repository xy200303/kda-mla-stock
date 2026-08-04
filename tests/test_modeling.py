from __future__ import annotations

import pytest
import torch

from kda_mla_stock.configuration import ModelConfig
from kda_mla_stock.models import StockForecaster, build_model


def small_model_config() -> ModelConfig:
    return ModelConfig(
        num_features=10,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        intermediate_size=64,
        kda_layers=[0],
        mla_layers=[1],
        kda_conv_kernel=3,
        qk_nope_head_dim=6,
        qk_rope_head_dim=2,
        value_head_dim=8,
        kv_lora_rank=8,
        dropout=0.0,
        attention_backend="torch",
    )


def test_hybrid_model_forward_and_backward() -> None:
    model = StockForecaster(small_model_config())
    features = torch.randn(3, 12, 10)
    predictions = model(features)
    assert predictions.shape == (3, 1)
    predictions.square().mean().backward()
    assert model.input_projection.weight.grad is not None


def test_model_accepts_right_padded_attention_mask() -> None:
    model = StockForecaster(small_model_config()).eval()
    features = torch.randn(2, 8, 10)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0]],
        dtype=torch.bool,
    )
    with torch.no_grad():
        predictions = model(features, mask)
    assert predictions.shape == (2, 1)
    assert torch.isfinite(predictions).all()


def test_optimized_model_loads_legacy_projection_weights() -> None:
    config = small_model_config()
    source = StockForecaster(config).eval()
    optimized_state = source.state_dict()
    legacy_state = {name: tensor.clone() for name, tensor in optimized_state.items()}

    for name in list(legacy_state):
        if name.endswith("qkvg_proj.weight"):
            weight = legacy_state.pop(name)
            prefix = name.removesuffix("qkvg_proj.weight")
            for projection, part in zip(("q", "k", "v", "g"), weight.chunk(4), strict=True):
                legacy_state[f"{prefix}{projection}_proj.weight"] = part
        elif name.endswith("qkvg_conv.conv.weight") or name.endswith("qkvg_conv.conv.bias"):
            parameter = name.rsplit(".", maxsplit=1)[-1]
            values = legacy_state.pop(name)
            prefix = name.removesuffix(f"qkvg_conv.conv.{parameter}")
            for projection, part in zip(("q", "k", "v", "g"), values.chunk(4), strict=True):
                legacy_state[f"{prefix}{projection}_conv.conv.{parameter}"] = part
        elif name.endswith("gate_up_proj.weight"):
            weight = legacy_state.pop(name)
            prefix = name.removesuffix("gate_up_proj.weight")
            gate, up = weight.chunk(2)
            legacy_state[f"{prefix}gate_proj.weight"] = gate
            legacy_state[f"{prefix}up_proj.weight"] = up

    restored = StockForecaster(config).eval()
    restored.load_state_dict(legacy_state, strict=True)
    features = torch.randn(2, 10, config.num_features)
    with torch.no_grad():
        expected = source(features)
        actual = restored(features)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("architecture", ["lstm", "gru", "transformer", "mlp"])
def test_baseline_model_factory_forward_and_backward(architecture: str) -> None:
    config = ModelConfig(
        architecture=architecture,
        num_features=10,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        intermediate_size=64,
        dropout=0.0,
    )
    model = build_model(config)
    features = torch.randn(3, 12, 10)
    predictions = model(features)
    assert predictions.shape == (3, 1)
    predictions.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

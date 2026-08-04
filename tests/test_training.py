from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from kda_mla_stock.configuration import ModelConfig, TrainingConfig
from kda_mla_stock.data import (
    build_window_datasets,
    engineer_features,
    fit_normalization_stats,
)
from kda_mla_stock.modeling import StockForecaster
from kda_mla_stock.training import load_model, train_model


def test_one_epoch_training_and_checkpoint_load(
    market_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        num_features=10,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=8,
        intermediate_size=32,
        kda_layers=[0],
        mla_layers=[],
        kda_conv_kernel=2,
        qk_nope_head_dim=6,
        qk_rope_head_dim=2,
        value_head_dim=8,
        kv_lora_rank=8,
        dropout=0.0,
        attention_backend="torch",
    )
    training_config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        sequence_length=8,
        horizon=3,
        train_end="2020-03-20",
        valid_end="2020-04-15",
        batch_size=32,
        epochs=1,
        mixed_precision="no",
        patience=1,
    )
    engineered = engineer_features(market_frame, training_config.horizon)
    stats = fit_normalization_stats(engineered, training_config.train_end)
    datasets = build_window_datasets(
        engineered,
        stats,
        training_config.sequence_length,
        training_config.train_end,
        training_config.valid_end,
    )
    model = StockForecaster(model_config)
    result = train_model(
        model,
        datasets,
        training_config,
        model_config,
        stats,
        requested_device="cpu",
    )
    checkpoint = tmp_path / "run" / "best.safetensors"
    assert result["epochs_completed"] == 1
    assert checkpoint.exists()
    assert (tmp_path / "run" / "tensorboard").exists()

    restored = StockForecaster(model_config)
    load_model(restored, checkpoint, torch.device("cpu"))
    assert torch.equal(
        restored.input_projection.weight,
        model.input_projection.weight,
    )

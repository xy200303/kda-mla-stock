from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from kda_mla_stock.configuration import ModelConfig, TrainingConfig
from kda_mla_stock.data import (
    FEATURE_COLUMNS,
    build_window_datasets,
    fit_normalization_stats,
    load_and_engineer_market_data,
)
from kda_mla_stock.modeling import StockForecaster, count_parameters
from kda_mla_stock.training import set_seed, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a KDA/MLA stock return forecaster")
    parser.add_argument("--model-config", default="configs/model-small.json")
    parser.add_argument("--train-config", default="configs/train.json")
    parser.add_argument("--data", default=None, help="Override data_path from the training config")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint file or output directory")
    parser.add_argument("--device", default=None, help="For example cuda, cuda:0, or cpu")
    args = parser.parse_args()

    model_config = ModelConfig.from_json(args.model_config)
    training_config = TrainingConfig.from_json(args.train_config)
    overrides = {}
    if args.data is not None:
        overrides["data_path"] = args.data
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if overrides:
        training_config = replace(training_config, **overrides)
        training_config.validate()
    if model_config.num_features != len(FEATURE_COLUMNS):
        raise ValueError(
            f"model expects {model_config.num_features} features, pipeline produces "
            f"{len(FEATURE_COLUMNS)}"
        )
    if not Path(training_config.data_path).exists():
        raise FileNotFoundError(
            f"data file not found: {training_config.data_path}. "
            "Run a data preparation script first."
        )

    print(f"loading and engineering features from {training_config.data_path}")
    frame = load_and_engineer_market_data(training_config.data_path, training_config.horizon)
    stats = fit_normalization_stats(frame, training_config.train_end)
    datasets = build_window_datasets(
        frame,
        stats,
        training_config.sequence_length,
        training_config.train_end,
        training_config.valid_end,
    )
    split_sizes = {name: len(dataset) for name, dataset in datasets.items()}
    print(f"dataset samples: {json.dumps(split_sizes)}")
    set_seed(training_config.seed)
    model = StockForecaster(model_config)
    print(f"trainable parameters: {count_parameters(model):,}")
    result = train_model(
        model,
        datasets,
        training_config,
        model_config,
        stats,
        resume_from=args.resume,
        requested_device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

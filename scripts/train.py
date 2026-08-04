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
from kda_mla_stock.modeling import build_model, count_parameters
from kda_mla_stock.training import set_seed, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a KDA/MLA stock return forecaster")
    parser.add_argument("--model-config", default="configs/model-small.json")
    parser.add_argument("--train-config", default="configs/train.json")
    parser.add_argument("--data", default=None, help="Override data_path from the training config")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--compile-mode",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        default=None,
    )
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
    if args.patience is not None:
        overrides["patience"] = args.patience
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.train_stride is not None:
        overrides["train_stride"] = args.train_stride
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.compile_mode is not None:
        overrides["compile_mode"] = args.compile_mode
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
        training_config.train_stride,
    )
    split_sizes = {name: len(dataset) for name, dataset in datasets.items()}
    print(
        f"dataset samples: {json.dumps(split_sizes)}, "
        f"training stride={training_config.train_stride}"
    )
    set_seed(training_config.seed)
    model = build_model(model_config)
    print(
        f"model: architecture={model_config.architecture}, "
        f"trainable parameters={count_parameters(model):,}"
    )
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

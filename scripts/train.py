from __future__ import annotations

import argparse
import json
from dataclasses import replace

from kda_mla_stock.core.config import TrainingConfig, load_model_config
from kda_mla_stock.orchestration import TrainRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural or traditional stock model")
    parser.add_argument("--model-config", default="configs/model-small.json")
    parser.add_argument("--train-config", default="configs/train.json")
    parser.add_argument("--data", default=None, help="Override data_path from the training config")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-stride", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--compile-mode",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        default=None,
    )
    parser.add_argument("--resume", default=None, help="Checkpoint file or output directory")
    parser.add_argument("--device", default=None, help="For example cuda, cuda:0, or cpu")
    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
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
    if args.log_interval is not None:
        overrides["log_interval"] = args.log_interval
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.compile_mode is not None:
        overrides["compile_mode"] = args.compile_mode
    if overrides:
        training_config = replace(training_config, **overrides)
        training_config.validate()
    result = TrainRunner(
        model_config,
        training_config,
        resume_from=args.resume,
        requested_device=args.device,
    ).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

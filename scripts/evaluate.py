from __future__ import annotations

import argparse
import json
from pathlib import Path

from kda_mla_stock.backtest import run_long_short_backtest, summarize_backtest
from kda_mla_stock.configuration import ModelConfig, TrainingConfig
from kda_mla_stock.data import (
    NormalizationStats,
    build_window_datasets,
    load_and_engineer_market_data,
)
from kda_mla_stock.metrics import daily_information_coefficients, evaluate_predictions
from kda_mla_stock.modeling import StockForecaster
from kda_mla_stock.training import (
    create_data_loader,
    load_model,
    predict_loader,
    resolve_device,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model and run a backtest")
    parser.add_argument("--checkpoint-dir", default="outputs/kda-mla-small")
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    model_config = ModelConfig.from_json(checkpoint_dir / "model_config.json")
    training_config = TrainingConfig.from_json(checkpoint_dir / "training_config.json")
    stats = NormalizationStats.from_json(checkpoint_dir / "normalization.json")
    data_path = args.data or training_config.data_path
    frame = load_and_engineer_market_data(data_path, training_config.horizon)
    datasets = build_window_datasets(
        frame,
        stats,
        training_config.sequence_length,
        training_config.train_end,
        training_config.valid_end,
    )
    dataset = datasets[args.split]
    if len(dataset) == 0:
        raise ValueError(f"{args.split} dataset has no samples")

    device = resolve_device(args.device)
    model = StockForecaster(model_config).to(device)
    load_model(model, checkpoint_dir / "best.safetensors", device)
    loader = create_data_loader(dataset, training_config, shuffle=False)
    loss, predictions = predict_loader(
        model,
        loader,
        device,
        training_config.mixed_precision,
        progress_description=f"Evaluate {args.split}",
    )
    prediction_metrics = evaluate_predictions(predictions)
    prediction_metrics["loss"] = loss
    backtest = run_long_short_backtest(
        predictions,
        top_fraction=training_config.top_fraction,
        transaction_cost_bps=training_config.transaction_cost_bps,
        rebalance_every=training_config.horizon,
    )
    backtest_metrics = summarize_backtest(
        backtest,
        periods_per_year=252.0 / training_config.horizon,
    )

    output_dir = Path(args.output_dir or checkpoint_dir / f"evaluation_{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    daily_information_coefficients(predictions).to_csv(output_dir / "daily_ic.csv", index=False)
    backtest.to_csv(output_dir / "backtest.csv", index=False)
    summary = {
        "split": args.split,
        "device": str(device),
        "prediction_metrics": prediction_metrics,
        "backtest_metrics": backtest_metrics,
    }
    write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

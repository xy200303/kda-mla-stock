from __future__ import annotations

import argparse
import json

from kda_mla_stock.engine import Valer
from kda_mla_stock.qlib_evaluation import QlibBacktestConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model and run a backtest")
    parser.add_argument("--checkpoint-dir", default="outputs/kda-mla-small")
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--qlib-provider-uri", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--qlib-region", choices=["cn", "us"], default="cn")
    parser.add_argument("--benchmark", default="SH000300")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n-drop", type=int, default=None)
    parser.add_argument("--hold-days", type=int, default=None)
    parser.add_argument("--deal-price", choices=["open", "close", "vwap"], default="open")
    parser.add_argument(
        "--skip-qlib",
        action="store_true",
        help="Only run the lightweight diagnostic backtest",
    )
    args = parser.parse_args()

    valer = Valer(args.checkpoint_dir, requested_device=args.device)
    training_config = valer.training_config
    qlib_config = None
    if not args.skip_qlib:
        qlib_config = QlibBacktestConfig(
            provider_uri=args.qlib_provider_uri,
            region=args.qlib_region,
            benchmark=args.benchmark,
            topk=args.topk,
            n_drop=args.n_drop or max(1, args.topk // training_config.horizon),
            hold_thresh=args.hold_days or training_config.horizon,
            deal_price=args.deal_price,
        )
    summary = valer.run(
        args.split,
        data_path=args.data,
        output_dir=args.output_dir,
        qlib_config=qlib_config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

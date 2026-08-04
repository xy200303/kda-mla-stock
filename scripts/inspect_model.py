from __future__ import annotations

import argparse
import json

from kda_mla_stock.configuration import ModelConfig
from kda_mla_stock.modeling import build_model, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Print model parameter counts")
    parser.add_argument("--model-config", default="configs/model-small.json")
    args = parser.parse_args()
    config = ModelConfig.from_json(args.model_config)
    model = build_model(config)
    report = {
        "architecture": config.architecture,
        "total": count_parameters(model),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

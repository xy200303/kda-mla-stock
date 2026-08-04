from __future__ import annotations

import argparse
import json

from kda_mla_stock.configuration import ModelConfig
from kda_mla_stock.modeling import StockForecaster, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Print model parameter counts")
    parser.add_argument("--model-config", default="configs/model-small.json")
    args = parser.parse_args()
    config = ModelConfig.from_json(args.model_config)
    model = StockForecaster(config)
    by_component = {
        "input": count_parameters(model.input_projection) + count_parameters(model.input_norm),
        "encoder": count_parameters(model.layers),
        "head": count_parameters(model.final_norm) + count_parameters(model.head),
        "total": count_parameters(model),
    }
    print(json.dumps(by_component, indent=2))


if __name__ == "__main__":
    main()

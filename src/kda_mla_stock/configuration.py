from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    num_features: int = 10
    hidden_size: int = 384
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    head_dim: int = 64
    intermediate_size: int = 1024
    kda_layers: list[int] = field(
        default_factory=lambda: [0, 1, 2, 4, 5, 6, 8, 9, 10]
    )
    mla_layers: list[int] = field(default_factory=lambda: [3, 7, 11])
    kda_conv_kernel: int = 4
    kda_gate_lower_bound: float | None = -5.0
    qk_nope_head_dim: int = 48
    qk_rope_head_dim: int = 16
    value_head_dim: int = 64
    kv_lora_rank: int = 96
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    dropout: float = 0.1
    attention_backend: str = "auto"
    num_targets: int = 1

    def validate(self) -> None:
        if self.num_features <= 0 or self.hidden_size <= 0 or self.num_hidden_layers <= 0:
            raise ValueError("feature, hidden, and layer sizes must be positive")
        if self.num_attention_heads <= 0 or self.head_dim <= 0:
            raise ValueError("attention dimensions must be positive")
        if len(set(self.kda_layers)) != len(self.kda_layers):
            raise ValueError("kda_layers contains duplicate indices")
        if len(set(self.mla_layers)) != len(self.mla_layers):
            raise ValueError("mla_layers contains duplicate indices")
        layer_indices = set(range(self.num_hidden_layers))
        kda_indices = set(self.kda_layers)
        mla_indices = set(self.mla_layers)
        if kda_indices & mla_indices:
            raise ValueError("KDA and MLA layer sets overlap")
        if kda_indices | mla_indices != layer_indices:
            raise ValueError("KDA and MLA layers must cover every encoder layer")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be even")
        if self.qk_nope_head_dim <= 0 or self.value_head_dim <= 0 or self.kv_lora_rank <= 0:
            raise ValueError("MLA dimensions must be positive")
        if self.num_targets <= 0:
            raise ValueError("num_targets must be positive")
        if self.attention_backend not in {"auto", "fla", "torch"}:
            raise ValueError("attention_backend must be auto, fla, or torch")

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        config = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        config.validate()
        return config

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelConfig:
        config = cls(**payload)
        config.validate()
        return config

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


@dataclass
class TrainingConfig:
    data_path: str = "data/market.csv"
    output_dir: str = "outputs/kda-mla-small"
    sequence_length: int = 256
    horizon: int = 5
    train_end: str = "2022-12-31"
    valid_end: str = "2023-12-31"
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    num_workers: int = 0
    patience: int = 8
    seed: int = 42
    transaction_cost_bps: float = 10.0
    top_fraction: float = 0.2

    def validate(self) -> None:
        if self.sequence_length <= 0 or self.horizon <= 0:
            raise ValueError("sequence_length and horizon must be positive")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if not 0.0 < self.top_fraction <= 0.5:
            raise ValueError("top_fraction must be in (0, 0.5]")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be no, fp16, or bf16")
        if self.train_end >= self.valid_end:
            raise ValueError("train_end must be before valid_end")

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingConfig:
        config = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        config.validate()
        return config

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

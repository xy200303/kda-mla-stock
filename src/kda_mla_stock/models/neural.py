from __future__ import annotations

import warnings
from importlib.util import find_spec
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from kda_mla_stock.configuration import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.normalized_shape = (hidden_size,)
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(hidden_states, self.normalized_shape, self.weight, self.eps)


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sequence = hidden_states.transpose(1, 2)
        sequence = F.pad(sequence, (self.kernel_size - 1, 0))
        return self.conv(sequence).transpose(1, 2)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, theta: float) -> None:
        super().__init__()
        frequencies = 1.0 / (
            theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.register_buffer("cosine_cache", torch.empty(0), persistent=False)
        self.register_buffer("sine_cache", torch.empty(0), persistent=False)

    def _update_cache(self, sequence_length: int, reference: torch.Tensor) -> None:
        cache_matches = (
            self.cosine_cache.shape[0] >= sequence_length
            and self.cosine_cache.device == reference.device
            and self.cosine_cache.dtype == reference.dtype
        )
        if cache_matches:
            return
        positions = torch.arange(sequence_length, device=reference.device, dtype=torch.float32)
        angles = torch.outer(positions, self.frequencies.float())
        embeddings = torch.cat((angles, angles), dim=-1)
        self.cosine_cache = embeddings.cos().to(reference.dtype)
        self.sine_cache = embeddings.sin().to(reference.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sequence_length = hidden_states.shape[-2]
        self._update_cache(sequence_length, hidden_states)
        cosine = self.cosine_cache[:sequence_length].view(1, 1, sequence_length, -1)
        sine = self.sine_cache[:sequence_length].view(1, 1, sequence_length, -1)
        return hidden_states * cosine + _rotate_half(hidden_states) * sine


def _upgrade_concatenated_parameter(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    new_name: str,
    legacy_names: tuple[str, ...],
) -> None:
    new_key = prefix + new_name
    legacy_keys = tuple(prefix + name for name in legacy_names)
    if new_key in state_dict or not all(key in state_dict for key in legacy_keys):
        return
    state_dict[new_key] = torch.cat([state_dict.pop(key) for key in legacy_keys], dim=0)


class GatedFeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        _upgrade_concatenated_parameter(
            state_dict,
            prefix,
            "gate_up_proj.weight",
            ("gate_proj.weight", "up_proj.weight"),
        )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention with an optional fla-core CUDA kernel."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        projection_size = config.num_attention_heads * config.head_dim
        self.qkvg_proj = nn.Linear(config.hidden_size, 4 * projection_size, bias=False)
        self.beta_proj = nn.Linear(config.hidden_size, config.num_attention_heads, bias=False)
        self.qkvg_conv = CausalDepthwiseConv1d(4 * projection_size, config.kda_conv_kernel)
        self.a_log = nn.Parameter(torch.zeros(config.num_attention_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        self.out_proj = nn.Linear(projection_size, config.hidden_size, bias=False)
        self._warned_fallback = False
        try:
            self.fla_available = find_spec("fla.ops.kda") is not None
        except ModuleNotFoundError:
            self.fla_available = False

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        _upgrade_concatenated_parameter(
            state_dict,
            prefix,
            "qkvg_proj.weight",
            ("q_proj.weight", "k_proj.weight", "v_proj.weight", "g_proj.weight"),
        )
        for parameter_name in ("weight", "bias"):
            _upgrade_concatenated_parameter(
                state_dict,
                prefix,
                f"qkvg_conv.conv.{parameter_name}",
                tuple(
                    f"{name}_conv.conv.{parameter_name}" for name in ("q", "k", "v", "g")
                ),
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _shape_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        return hidden_states.view(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.config.head_dim,
        )

    def _can_use_fla(
        self,
        projected_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> bool:
        if self.config.attention_backend == "torch":
            return False
        if not projected_states.is_cuda:
            return False
        if projected_states.dtype not in {torch.float16, torch.bfloat16}:
            return False
        if attention_mask is not None and not bool(attention_mask.all()):
            return False
        if not self.fla_available:
            if self.config.attention_backend == "fla":
                raise RuntimeError("attention_backend=fla requires fla-core") from None
            return False
        return True

    def _fla_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        from fla.ops.kda import chunk_kda

        output, _ = chunk_kda(
            query,
            key,
            value,
            gate,
            beta,
            A_log=self.a_log,
            dt_bias=self.dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=self.config.kda_gate_lower_bound is not None,
            lower_bound=self.config.kda_gate_lower_bound,
        )
        return output

    def _torch_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self._warned_fallback and query.is_cuda:
            warnings.warn(
                "KDA is using the slow PyTorch path on CUDA; install fla-core or set "
                "attention_backend=fla to fail fast.",
                stacklevel=2,
            )
            self._warned_fallback = True
        query = F.normalize(query.float(), dim=-1).to(value.dtype)
        key = F.normalize(key.float(), dim=-1).to(value.dtype)
        beta = torch.sigmoid(raw_beta)
        a = self.a_log.exp().view(1, -1, 1)
        dt_bias = self.dt_bias.view(
            1,
            self.config.num_attention_heads,
            self.config.head_dim,
        )
        if self.config.kda_gate_lower_bound is not None:
            gate = self.config.kda_gate_lower_bound * torch.sigmoid(a * (raw_gate + dt_bias))
        else:
            gate = -a * F.softplus(raw_gate + dt_bias)

        batch_size, sequence_length, num_heads, key_dim = query.shape
        value_dim = value.shape[-1]
        state = value.new_zeros(batch_size, num_heads, key_dim, value_dim)
        outputs = []
        for position in range(sequence_length):
            token_mask = None
            if attention_mask is not None:
                token_mask = attention_mask[:, position].bool().view(batch_size, 1, 1, 1)
            decayed_state = state * gate[:, position].exp().unsqueeze(-1)
            current_key = key[:, position]
            current_value = value[:, position]
            prediction = torch.einsum("bhkv,bhk->bhv", decayed_state, current_key)
            error = current_value - prediction
            update = torch.einsum("bhk,bhv->bhkv", current_key, error)
            update = update * beta[:, position].unsqueeze(-1).unsqueeze(-1)
            next_state = decayed_state + update
            state = (
                torch.where(token_mask, next_state, state)
                if token_mask is not None
                else next_state
            )
            output = torch.einsum("bhk,bhkv->bhv", query[:, position], state)
            if token_mask is not None:
                output = output * token_mask.squeeze(-1)
            outputs.append(output)
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected = self.qkvg_conv(self.qkvg_proj(hidden_states))
        query, key, value, gate = (
            self._shape_heads(states) for states in projected.chunk(4, dim=-1)
        )
        beta = self.beta_proj(hidden_states)
        if self._can_use_fla(query, attention_mask):
            output = self._fla_forward(query, key, value, gate, beta)
        else:
            output = self._torch_forward(query, key, value, gate, beta, attention_mask)
        output = output.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        return self.out_proj(output)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        query_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * query_dim,
            bias=False,
        )
        self.kv_down = nn.Linear(config.hidden_size, config.kv_lora_rank, bias=False)
        self.kv_up = nn.Linear(
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_nope_head_dim + config.value_head_dim),
            bias=False,
        )
        self.k_rope_proj = nn.Linear(config.hidden_size, config.qk_rope_head_dim, bias=False)
        self.rotary = RotaryEmbedding(config.qk_rope_head_dim, config.rope_theta)
        self.out_proj = nn.Linear(
            config.num_attention_heads * config.value_head_dim,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        num_heads = self.config.num_attention_heads
        query_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        query = self.q_proj(hidden_states).view(batch_size, sequence_length, num_heads, query_dim)
        query_nope, query_rope = query.split(
            [self.config.qk_nope_head_dim, self.config.qk_rope_head_dim],
            dim=-1,
        )
        latent_kv = self.kv_down(hidden_states)
        key_value = self.kv_up(latent_kv).view(
            batch_size,
            sequence_length,
            num_heads,
            self.config.qk_nope_head_dim + self.config.value_head_dim,
        )
        key_nope, value = key_value.split(
            [self.config.qk_nope_head_dim, self.config.value_head_dim],
            dim=-1,
        )
        key_rope = self.k_rope_proj(hidden_states).unsqueeze(2).expand(-1, -1, num_heads, -1)
        query_rope = self.rotary(query_rope.transpose(1, 2))
        key_rope = self.rotary(key_rope.transpose(1, 2))
        query = torch.cat((query_nope.transpose(1, 2), query_rope), dim=-1)
        key = torch.cat((key_nope.transpose(1, 2), key_rope), dim=-1)
        value = value.transpose(1, 2)

        dropout = self.config.dropout if self.training else 0.0
        if attention_mask is None or bool(attention_mask.all()):
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=dropout,
                is_causal=True,
            )
        else:
            causal = torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=hidden_states.device,
            ).tril()
            allowed = causal.view(1, 1, sequence_length, sequence_length)
            allowed = allowed & attention_mask[:, None, None, :].bool()
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=allowed,
                dropout_p=dropout,
            )
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return self.out_proj(output)


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if layer_index in config.kda_layers:
            self.attention: nn.Module = KimiDeltaAttention(config)
        else:
            self.attention = MultiHeadLatentAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.feed_forward = GatedFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.dropout(
            self.attention(self.attention_norm(hidden_states), attention_mask)
        )
        hidden_states = hidden_states + self.dropout(
            self.feed_forward(self.ffn_norm(hidden_states))
        )
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class StockForecaster(nn.Module):
    """Causal time-series encoder that predicts future returns from OHLCV features."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.input_projection = nn.Linear(config.num_features, config.hidden_size)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            EncoderLayer(config, layer_index) for layer_index in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_targets),
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.config.num_features:
            raise ValueError(
                f"features must have shape [batch, sequence, {self.config.num_features}]"
            )
        if attention_mask is not None and attention_mask.shape != features.shape[:2]:
            raise ValueError("attention_mask must match the first two feature dimensions")
        hidden_states = self.input_norm(self.input_projection(features))
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.final_norm(hidden_states)

        if attention_mask is None:
            pooled = hidden_states[:, -1]
        else:
            last_indices = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
            batch_indices = torch.arange(features.shape[0], device=features.device)
            pooled = hidden_states[batch_indices, last_indices]
        return self.head(pooled)


class RecurrentForecaster(nn.Module):
    """LSTM or GRU baseline using the same input windows and prediction head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        if config.architecture not in {"lstm", "gru"}:
            raise ValueError("RecurrentForecaster requires architecture=lstm or gru")
        self.config = config
        recurrent_class = nn.LSTM if config.architecture == "lstm" else nn.GRU
        self.recurrent = recurrent_class(
            input_size=config.num_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            batch_first=True,
            dropout=config.dropout if config.num_hidden_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_targets),
        )

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_features(features, attention_mask, self.config.num_features)
        hidden_states, _ = self.recurrent(features)
        pooled = _last_valid_state(hidden_states, attention_mask)
        return self.head(pooled)


class TransformerForecaster(nn.Module):
    """Causal Transformer encoder baseline."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        if config.architecture != "transformer":
            raise ValueError("TransformerForecaster requires architecture=transformer")
        self.config = config
        self.input_projection = nn.Linear(config.num_features, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_hidden_layers,
            norm=nn.LayerNorm(config.hidden_size),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(config.hidden_size, config.num_targets)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_features(features, attention_mask, self.config.num_features)
        hidden_states = self.input_projection(features)
        hidden_states = hidden_states + _sinusoidal_positions(hidden_states)
        sequence_length = features.shape[1]
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=features.device,
        ).triu(1)
        padding_mask = None if attention_mask is None else ~attention_mask.bool()
        hidden_states = self.encoder(
            hidden_states,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return self.head(_last_valid_state(hidden_states, attention_mask))


class TemporalMLPForecaster(nn.Module):
    """Low-cost baseline over the last, mean, and volatility states of a window."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        if config.architecture != "mlp":
            raise ValueError("TemporalMLPForecaster requires architecture=mlp")
        self.config = config
        layers: list[nn.Module] = []
        input_size = config.num_features * 3
        for layer_index in range(config.num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(
                        input_size if layer_index == 0 else config.hidden_size,
                        config.hidden_size,
                    ),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
        layers.append(nn.Linear(config.hidden_size, config.num_targets))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_features(features, attention_mask, self.config.num_features)
        last = _last_valid_state(features, attention_mask)
        if attention_mask is None:
            mean = features.mean(dim=1)
            std = features.std(dim=1, unbiased=False)
        else:
            weights = attention_mask.unsqueeze(-1).to(features.dtype)
            count = weights.sum(dim=1).clamp_min(1.0)
            mean = (features * weights).sum(dim=1) / count
            variance = ((features - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count
            std = variance.sqrt()
        return self.network(torch.cat((last, mean, std), dim=-1))


def _validate_features(
    features: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_features: int,
) -> None:
    if features.ndim != 3 or features.shape[-1] != num_features:
        raise ValueError(f"features must have shape [batch, sequence, {num_features}]")
    if attention_mask is not None and attention_mask.shape != features.shape[:2]:
        raise ValueError("attention_mask must match the first two feature dimensions")


def _last_valid_state(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states[:, -1]
    last_indices = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, last_indices]


def _sinusoidal_positions(hidden_states: torch.Tensor) -> torch.Tensor:
    sequence_length, hidden_size = hidden_states.shape[1:]
    positions = torch.arange(sequence_length, device=hidden_states.device).float().unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, hidden_size, 2, device=hidden_states.device).float()
        * (-torch.log(torch.tensor(10_000.0, device=hidden_states.device)) / hidden_size)
    )
    encoding = hidden_states.new_zeros(sequence_length, hidden_size)
    encoding[:, 0::2] = torch.sin(positions * frequencies).to(hidden_states.dtype)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: hidden_size // 2]).to(
        hidden_states.dtype
    )
    return encoding.unsqueeze(0)


def build_model(config: ModelConfig) -> nn.Module:
    if config.architecture == "kda_mla":
        return StockForecaster(config)
    if config.architecture in {"lstm", "gru"}:
        return RecurrentForecaster(config)
    if config.architecture == "transformer":
        return TransformerForecaster(config)
    if config.architecture == "mlp":
        return TemporalMLPForecaster(config)
    raise ValueError(f"unsupported architecture: {config.architecture}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

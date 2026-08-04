from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import nn

from kda_mla_stock.configuration import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


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


def apply_rotary(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    rotary_dim = hidden_states.shape[-1]
    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                rotary_dim,
                2,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            / rotary_dim
        )
    )
    angles = position_ids.float().unsqueeze(-1) * frequencies
    embeddings = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    cosine = embeddings.cos().to(hidden_states.dtype)
    sine = embeddings.sin().to(hidden_states.dtype)
    return hidden_states * cosine + _rotate_half(hidden_states) * sine


class GatedFeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention with an optional fla-core CUDA kernel."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        projection_size = config.num_attention_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.g_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.beta_proj = nn.Linear(config.hidden_size, config.num_attention_heads, bias=False)
        self.q_conv = CausalDepthwiseConv1d(projection_size, config.kda_conv_kernel)
        self.k_conv = CausalDepthwiseConv1d(projection_size, config.kda_conv_kernel)
        self.v_conv = CausalDepthwiseConv1d(projection_size, config.kda_conv_kernel)
        self.g_conv = CausalDepthwiseConv1d(projection_size, config.kda_conv_kernel)
        self.a_log = nn.Parameter(torch.zeros(config.num_attention_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        self.out_proj = nn.Linear(projection_size, config.hidden_size, bias=False)
        self._warned_fallback = False

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
        try:
            from fla.ops.kda import chunk_kda  # noqa: F401
        except ImportError:
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
        if not self._warned_fallback and query.shape[1] > 512:
            warnings.warn(
                "KDA is using the slow PyTorch path; install fla-core on CUDA for training.",
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
        query = self._shape_heads(self.q_conv(self.q_proj(hidden_states)))
        key = self._shape_heads(self.k_conv(self.k_proj(hidden_states)))
        value = self._shape_heads(self.v_conv(self.v_proj(hidden_states)))
        gate = self._shape_heads(self.g_conv(self.g_proj(hidden_states)))
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
        position_ids = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
        query_rope = apply_rotary(query_rope.transpose(1, 2), position_ids, self.config.rope_theta)
        key_rope = apply_rotary(key_rope.transpose(1, 2), position_ids, self.config.rope_theta)
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


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

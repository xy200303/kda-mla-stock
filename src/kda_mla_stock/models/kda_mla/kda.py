from __future__ import annotations

import warnings
from importlib.util import find_spec
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from kda_mla_stock.core.config import ModelConfig
from kda_mla_stock.models.kda_mla.common import (
    CausalDepthwiseConv1d,
    upgrade_concatenated_parameter,
)


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
        upgrade_concatenated_parameter(
            state_dict,
            prefix,
            "qkvg_proj.weight",
            ("q_proj.weight", "k_proj.weight", "v_proj.weight", "g_proj.weight"),
        )
        for parameter_name in ("weight", "bias"):
            upgrade_concatenated_parameter(
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
        dt_bias = self.dt_bias.view(1, self.config.num_attention_heads, self.config.head_dim)
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

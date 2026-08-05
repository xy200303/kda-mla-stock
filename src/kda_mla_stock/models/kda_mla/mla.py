from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kda_mla_stock.core.config import ModelConfig
from kda_mla_stock.models.kda_mla.common import RotaryEmbedding


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
        query = self.q_proj(hidden_states).view(
            batch_size, sequence_length, num_heads, query_dim
        )
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

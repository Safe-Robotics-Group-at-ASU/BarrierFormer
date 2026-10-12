"""GPT-2-style causal transformer backbone in JAX/Flax."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from .utils import get_act_from_str, default_nn_init


@dataclass(frozen=True)
class TransformerConfig:
    """Hyperparameters for the causal transformer backbone."""

    hidden_dim: int
    num_heads: int
    num_layers: int
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    max_seq_len: int = 25
    obs_dim: int = 70
    action_dim: int = 2
    activation: str = "gelu"


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention with causal mask."""
    hidden_dim: int
    num_heads: int
    dropout_rate: float

    def setup(self) -> None:
        """Create the QKV and output projections."""
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        self.head_dim = self.hidden_dim // self.num_heads

        self.q_proj   = nn.Dense(self.hidden_dim, kernel_init=default_nn_init())
        self.k_proj   = nn.Dense(self.hidden_dim, kernel_init=default_nn_init())
        self.v_proj   = nn.Dense(self.hidden_dim, kernel_init=default_nn_init())
        self.out_proj = nn.Dense(self.hidden_dim, kernel_init=default_nn_init())
        self.attn_drop = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        x: jnp.ndarray,
        causal_mask: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Run multi-head causal self-attention."""
        batch, seq_len, _ = x.shape

        def _project_and_split(proj):
            return proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        Q = _project_and_split(self.q_proj).transpose(0, 2, 1, 3)
        K = _project_and_split(self.k_proj).transpose(0, 2, 1, 3)
        V = _project_and_split(self.v_proj).transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn_logits = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) * scale

        attn_logits = jnp.where(
            causal_mask[None, None, :, :] == 0,
            jnp.finfo(attn_logits.dtype).min,
            attn_logits,
        )

        attn_weights = nn.softmax(attn_logits, axis=-1)
        attn_weights = self.attn_drop(attn_weights, deterministic=deterministic)

        out = jnp.matmul(attn_weights, V)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.hidden_dim)

        return self.out_proj(out)

class PositionwiseFFN(nn.Module):
    """Two-layer position-wise FFN: Linear -> activation -> Linear."""

    hidden_dim: int
    mlp_dim: int
    activation: str
    dropout_rate: float

    def setup(self) -> None:
        """Create the two linear layers."""
        self.fc1  = nn.Dense(self.mlp_dim,    kernel_init=default_nn_init())
        self.fc2  = nn.Dense(self.hidden_dim, kernel_init=default_nn_init())
        self.drop = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Apply FFN with optional dropout on the output."""
        act = get_act_from_str(self.activation)
        x = act(self.fc1(x))
        x = self.fc2(x)
        x = self.drop(x, deterministic=deterministic)
        return x


class CausalTransformerBlock(nn.Module):
    """Single GPT-2-style transformer block with pre-LayerNorm."""

    hidden_dim: int
    num_heads: int
    mlp_ratio: float
    dropout_rate: float
    activation: str

    def setup(self) -> None:
        """Create the layer norms, attention and FFN."""
        mlp_dim = int(self.mlp_ratio * self.hidden_dim)

        self.ln1  = nn.LayerNorm()
        self.attn = MultiHeadSelfAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        self.ln2  = nn.LayerNorm()
        self.ffn  = PositionwiseFFN(
            hidden_dim=self.hidden_dim,
            mlp_dim=mlp_dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        causal_mask: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Apply one transformer block with pre-LayerNorm and residual connections."""
        x = x + self.attn(self.ln1(x), causal_mask, deterministic)
        x = x + self.ffn(self.ln2(x), deterministic)
        return x


class CausalTransformer(nn.Module):
    """GPT-2-style causal transformer that encodes a history window H_t."""

    config: TransformerConfig

    def setup(self) -> None:
        """Create the input projection, position embedding and blocks."""
        cfg = self.config

        self.obs_embed = nn.Dense(cfg.hidden_dim, kernel_init=default_nn_init())
        self.act_embed = nn.Dense(cfg.hidden_dim, kernel_init=default_nn_init())

        self.pos_embed = nn.Embed(cfg.max_seq_len, cfg.hidden_dim)

        self.blocks = [
            CausalTransformerBlock(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout_rate=cfg.dropout_rate,
                activation=cfg.activation,
            )
            for _ in range(cfg.num_layers)
        ]

        self.ln_f = nn.LayerNorm()

    def __call__(
        self,
        obs_seq: jnp.ndarray,
        action_seq: jnp.ndarray,
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Encode the history window into a sequence of hidden states."""
        batch = obs_seq.shape[0]
        T = action_seq.shape[1]
        seq_len = 2 * T + 1

        obs_emb = self.obs_embed(obs_seq)
        act_emb = self.act_embed(action_seq)

        interleaved = jnp.zeros(
            (batch, seq_len, self.config.hidden_dim), dtype=obs_emb.dtype
        )
        interleaved = interleaved.at[:, 0::2, :].set(obs_emb)
        interleaved = interleaved.at[:, 1::2, :].set(act_emb)

        positions = jnp.arange(seq_len)
        pos_emb   = self.pos_embed(positions)
        x = interleaved + pos_emb[None, :, :]

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

        for block in self.blocks:
            x = block(x, causal_mask, deterministic)

        x = self.ln_f(x)

        z_t = x[:, -1, :]

        return x, z_t

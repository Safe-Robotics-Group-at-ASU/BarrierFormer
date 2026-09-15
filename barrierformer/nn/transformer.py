"""
GPT-2-style causal transformer backbone in JAX/Flax.

Sequence layout (seq_len = 2*T + 1):
    pos 0     pos 1     pos 2     pos 3  ...  pos 2T-2  pos 2T-1  pos 2T
    o_{t-T}  u_{t-T}  o_{t-T+1} u_{t-T+1}   o_{t-1}   u_{t-1}    o_t

Observations occupy even positions (0, 2, ..., 2T).
Actions occupy odd positions (1, 3, ..., 2T-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from .utils import get_act_from_str, default_nn_init


# config class

@dataclass(frozen=True)
class TransformerConfig:
    """Hyperparameters for the causal transformer backbone.

    Attributes
    ----------
    hidden_dim    : Model width. Must be divisible by num_heads.
    num_heads     : Number of attention heads.
    num_layers    : Number of stacked transformer blocks.
    mlp_ratio     : FFN hidden dim = mlp_ratio * hidden_dim.
    dropout_rate  : Dropout probability (applied only when deterministic=False).
    max_seq_len   : Maximum sequence length = 2*T + 1.
    obs_dim       : Dimensionality of a single observation o_t.
    action_dim    : Dimensionality of a single action u_t.
    activation    : Activation function name ('gelu', 'relu', 'tanh', ...).
    """

    hidden_dim: int
    num_heads: int
    num_layers: int
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1 #when the dropout is applied, it is applied to both attention weights and FFN output
    max_seq_len: int = 25        # 2*T+1 for default T=12 
    obs_dim: int = 70
    action_dim: int = 2
    activation: str = "gelu" # DIFFERENCE fROM RELU IS THAT GELU IS SMOOTHER AND HAS NONZERO GRADIENTS FOR NEGATIVE INPUTS, WHICH CAN HELP WITH TRAINING STABILITY


# Multi-head causal self-attention

class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention with causal mask.

    The causal mask is passed in from the enclosing block so it is built once and reused across every layer.

    Attributes
    ----------
    hidden_dim   : Total model width (= num_heads * head_dim).
    num_heads    : Number of attention heads.
    dropout_rate : Attention-weight dropout probability.
    """
    hidden_dim: int
    num_heads: int
    dropout_rate: float

    def setup(self) -> None:
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
        x: jnp.ndarray,          # (batch, seq_len, hidden_dim)
        causal_mask: jnp.ndarray, # (seq_len, seq_len) lower-triangular {0,1}
        deterministic: bool = True,
    ) -> jnp.ndarray:             # (batch, seq_len, hidden_dim)
        """Run multi-head causal self-attention.

        Parameters
        ----------
        x            : Input hidden states. Shape: (batch, seq_len, hidden_dim).
        causal_mask  : Lower-triangular binary mask. Shape: (seq_len, seq_len).
        deterministic: If False, applies dropout to attention weights.

        Returns
        -------
        out : Attended output. Shape: (batch, seq_len, hidden_dim).
        """
        batch, seq_len, _ = x.shape

        # Project and split into heads: (batch, seq, num_heads, head_dim)
        def _project_and_split(proj):
            return proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        Q = _project_and_split(self.q_proj).transpose(0, 2, 1, 3)  # (B, H, S, D)
        K = _project_and_split(self.k_proj).transpose(0, 2, 1, 3)
        V = _project_and_split(self.v_proj).transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5 # Scale factor for stable gradients (Vaswani et al. 2017)
        # (batch, heads, seq, seq)
        attn_logits = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) * scale

        # Mask: where mask==0 (future positions), set logit to -inf
        attn_logits = jnp.where(
            causal_mask[None, None, :, :] == 0,
            jnp.finfo(attn_logits.dtype).min,
            attn_logits,
        )

        attn_weights = nn.softmax(attn_logits, axis=-1)          # (B, H, S, S)
        attn_weights = self.attn_drop(attn_weights, deterministic=deterministic)

        # (batch, heads, seq, head_dim) -> (batch, seq, hidden_dim)
        out = jnp.matmul(attn_weights, V)                        # (B, H, S, D)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.hidden_dim)

        return self.out_proj(out)

# Position-wise feed-forward network
class PositionwiseFFN(nn.Module):
    """Two-layer position-wise FFN: Linear -> activation -> Linear.

    Attributes
    ----------
    hidden_dim   : Input and output width.
    mlp_dim      : Inner width (= mlp_ratio * hidden_dim from config).
    activation   : Activation function string ('gelu', 'relu', 'tanh', ...).
    dropout_rate : Applied to FFN output (only when deterministic=False).
    """

    hidden_dim: int
    mlp_dim: int
    activation: str
    dropout_rate: float

    def setup(self) -> None:
        self.fc1  = nn.Dense(self.mlp_dim,    kernel_init=default_nn_init())
        self.fc2  = nn.Dense(self.hidden_dim, kernel_init=default_nn_init())
        self.drop = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        x: jnp.ndarray,       # (batch, seq_len, hidden_dim)
        deterministic: bool = True,
    ) -> jnp.ndarray:         # (batch, seq_len, hidden_dim)
        """Apply FFN with optional dropout on the output."""
        act = get_act_from_str(self.activation)
        x = act(self.fc1(x))
        x = self.fc2(x)
        x = self.drop(x, deterministic=deterministic)
        return x


# Transformer block (Pre-LayerNorm)
class CausalTransformerBlock(nn.Module):
    """Single GPT-2-style transformer block with pre-LayerNorm.
    Architecture (Pre-LN, following GPT-2):
        residual = x
        x = LN(x)
        x = residual + MHSA(x, mask)
        residual = x
        x = LN(x)
        x = residual + FFN(x)

    Attributes
    ----------
    hidden_dim   : Model width.
    num_heads    : Number of attention heads.
    mlp_ratio    : FFN inner width ratio.
    dropout_rate : Dropout for attention and FFN (when deterministic=False).
    activation   : FFN activation function string.
    """

    hidden_dim: int
    num_heads: int
    mlp_ratio: float
    dropout_rate: float
    activation: str

    def setup(self) -> None:
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
        x: jnp.ndarray,          # (batch, seq_len, hidden_dim)
        causal_mask: jnp.ndarray, # (seq_len, seq_len)
        deterministic: bool = True,
    ) -> jnp.ndarray:             # (batch, seq_len, hidden_dim)
        """Apply one transformer block with pre-LayerNorm and residual connections."""
        # Attention sub-layer
        x = x + self.attn(self.ln1(x), causal_mask, deterministic)
        # FFN sub-layer
        x = x + self.ffn(self.ln2(x), deterministic)
        return x


# Full causal transformer
class CausalTransformer(nn.Module):
    """GPT-2-style causal transformer that encodes a history window H_t.

    Input format:
        obs_seq    : (batch, T+1, obs_dim)   — T past obs + current obs
        action_seq : (batch, T, action_dim)  — T past actions

    The two sequences are projected to hidden_dim, then interleaved into a
    single token stream of length 2*T+1:
        [o_{t-T}, u_{t-T}, o_{t-T+1}, u_{t-T+1}, ..., o_{t-1}, u_{t-1}, o_t]

    A shared positional embedding is indexed by the absolute position in this
    interleaved sequence.

    Output:
        all_hidden_states : (batch, 2*T+1, hidden_dim)
        z_t               : (batch, hidden_dim)  — last-token latent

    Attributes
    ----------
    config : TransformerConfig frozen dataclass with all hyperparameters.
    """

    config: TransformerConfig

    def setup(self) -> None:
        cfg = self.config

        # Separate input projection for observations and actions
        self.obs_embed = nn.Dense(cfg.hidden_dim, kernel_init=default_nn_init())
        self.act_embed = nn.Dense(cfg.hidden_dim, kernel_init=default_nn_init())

        # Learned positional embedding table: one entry per absolute position
        self.pos_embed = nn.Embed(cfg.max_seq_len, cfg.hidden_dim)

        # Stack of transformer blocks (Python list — Flax names them blocks_0, blocks_1, ...)
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

        # Final layer norm (GPT-2 style: applied before the output projection)
        self.ln_f = nn.LayerNorm()

    def __call__(
        self,
        obs_seq: jnp.ndarray,      # (batch, T+1, obs_dim)
        action_seq: jnp.ndarray,   # (batch, T, action_dim)
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Encode the history window into a sequence of hidden states.
        Parameters ----------
        obs_seq       : Observation history + current obs. Shape: (batch, T+1, obs_dim).
        action_seq    : Past action history (T actions). Shape: (batch, T, action_dim).
        deterministic : If False, enables dropout during attention and FFN.

        Returns -------
        all_hidden_states : Full sequence of hidden states. Shape: (batch, 2*T+1, hidden_dim).
        z_t               : Last-token latent representation. Shape: (batch, hidden_dim).
        """
        batch = obs_seq.shape[0]
        T = action_seq.shape[1]           # number of past action steps
        seq_len = 2 * T + 1               # interleaved sequence length

        # ── 1. Embed each token type ─────────────────────────────────────────
        obs_emb = self.obs_embed(obs_seq)      # (batch, T+1, hidden_dim)
        act_emb = self.act_embed(action_seq)   # (batch, T,   hidden_dim)

        # ── 2. Interleave into one sequence ──────────────────────────────────
        # Layout: obs at even indices, actions at odd indices
        #   even: 0, 2, ..., 2T   ← T+1 obs
        #   odd : 1, 3, ..., 2T-1 ← T   actions
        interleaved = jnp.zeros(
            (batch, seq_len, self.config.hidden_dim), dtype=obs_emb.dtype
        )
        interleaved = interleaved.at[:, 0::2, :].set(obs_emb)   # even positions
        interleaved = interleaved.at[:, 1::2, :].set(act_emb)   # odd  positions

        # ── 3. Add positional embeddings ─────────────────────────────────────
        positions = jnp.arange(seq_len)                          # (seq_len,)
        pos_emb   = self.pos_embed(positions)                    # (seq_len, hidden_dim)
        x = interleaved + pos_emb[None, :, :]                   # (batch, seq_len, hidden_dim)

        # ── 4. Build causal mask (lower-triangular, stays constant) ──────────
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

        # ── 5. Pass through transformer blocks ───────────────────────────────
        for block in self.blocks:
            x = block(x, causal_mask, deterministic)

        # ── 6. Final layer norm ───────────────────────────────────────────────
        x = self.ln_f(x)                                         # (batch, seq_len, hidden_dim)

        # Last token (position 2T = current observation o_t) is the latent
        z_t = x[:, -1, :]                                        # (batch, hidden_dim)

        return x, z_t

"""Residual action-correction head."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from .utils import get_act_from_str, default_nn_init


class ActionHead(nn.Module):
    """MLP that maps the transformer latent z_t to a residual action delta_u."""

    hidden_dim: int
    action_dim: int
    head_hidden_dim: int = 64
    num_layers: int = 2
    activation: str = "tanh"

    def setup(self) -> None:
        """Create the MLP layers."""
        self.hidden_layers = [
            nn.Dense(self.head_hidden_dim, kernel_init=default_nn_init())
            for _ in range(self.num_layers)
        ]
        self.ln = nn.LayerNorm()
        self.out_proj = nn.Dense(self.action_dim, kernel_init=default_nn_init())

    def __call__(self, z_t: jnp.ndarray) -> jnp.ndarray:
        """Compute residual action correction from the transformer latent."""
        act = get_act_from_str(self.activation)

        x = self.ln(z_t)
        for layer in self.hidden_layers:
            x = act(layer(x))
        return self.out_proj(x)

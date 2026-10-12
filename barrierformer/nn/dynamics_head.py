"""State-dynamics prediction head."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from .utils import get_act_from_str, default_nn_init


class DynamicsHead(nn.Module):
    """MLP that predicts state increment delta_x given z_t and u_applied."""

    hidden_dim: int
    action_dim: int
    state_dim: int
    head_hidden_dim: int = 128
    num_layers: int = 3
    activation: str = "gelu"

    def setup(self) -> None:
        """Create the MLP layers."""
        self.ln = nn.LayerNorm()

        in_dim = self.hidden_dim + self.action_dim
        self.first_layer = nn.Dense(self.head_hidden_dim, kernel_init=default_nn_init())

        self.hidden_layers = [
            nn.Dense(self.head_hidden_dim, kernel_init=default_nn_init())
            for _ in range(self.num_layers - 1)
        ]

        self.out_proj = nn.Dense(self.state_dim, kernel_init=default_nn_init())

    def __call__(
        self,
        z_t: jnp.ndarray,
        u_applied: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predict the state increment delta_x_hat."""
        act = get_act_from_str(self.activation)

        x = jnp.concatenate([z_t, u_applied], axis=-1)
        x = self.ln(x)

        x = act(self.first_layer(x))
        for layer in self.hidden_layers:
            x = act(layer(x))

        return self.out_proj(x)

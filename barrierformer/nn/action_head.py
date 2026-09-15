"""
Residual action-correction head.

Takes the last-token latent z_t from the causal transformer and outputs
delta_u, the residual correction to add to the nominal action u_nom:

    u_applied = u_nom + delta_u

Architecture:
    LayerNorm(z_t)
    -> [Linear(hidden_dim -> head_hidden_dim) -> activation] x num_layers
    -> Linear(head_hidden_dim -> action_dim)

No dropout. The final output is unbounded — the caller clips if needed.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from .utils import get_act_from_str, default_nn_init


class ActionHead(nn.Module):
    """MLP that maps the transformer latent z_t to a residual action delta_u.

    Attributes
    ----------
    hidden_dim      : Must match the transformer's hidden_dim (input width).
    action_dim      : Dimensionality of the action space.
    head_hidden_dim : Width of each hidden layer in the head MLP.
    num_layers      : Number of hidden layers (not counting the output layer).
    activation      : Activation function string ('tanh', 'relu', 'gelu', ...).
    """

    hidden_dim: int
    action_dim: int
    head_hidden_dim: int = 64
    num_layers: int = 2
    activation: str = "tanh"

    def setup(self) -> None:
        # Hidden layers
        self.hidden_layers = [
            nn.Dense(self.head_hidden_dim, kernel_init=default_nn_init())
            for _ in range(self.num_layers)
        ]
        # Input layer norm
        self.ln = nn.LayerNorm()
        # Output projection (no activation — unbounded residual)
        self.out_proj = nn.Dense(self.action_dim, kernel_init=default_nn_init())

    def __call__(self, z_t: jnp.ndarray) -> jnp.ndarray:
        """Compute residual action correction from the transformer latent.

        Parameters
        ----------
        z_t : Transformer last-token latent. Shape: (batch, hidden_dim).

        Returns
        -------
        delta_u : Residual action correction. Shape: (batch, action_dim).
                  Caller computes u_applied = u_nom + delta_u.
        """
        act = get_act_from_str(self.activation)

        x = self.ln(z_t)
        for layer in self.hidden_layers:
            x = act(layer(x))
        return self.out_proj(x)

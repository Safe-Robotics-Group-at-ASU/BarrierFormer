"""
State-dynamics prediction head.

Takes the transformer latent z_t and the full applied action u_applied,
and predicts the state increment delta_x_hat:

    input       = concat([z_t, u_applied])       # (batch, hidden_dim + action_dim)
    delta_x_hat = DynamicsHead(input)             # (batch, state_dim)
    x_hat_next  = x_t + delta_x_hat              # caller computes this

Architecture:
    LayerNorm(concat([z_t, u_applied]))
    -> [Linear -> activation] x num_layers
    -> Linear(-> state_dim)

Output is unbounded — delta_x_hat is a raw increment.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from .utils import get_act_from_str, default_nn_init


class DynamicsHead(nn.Module):
    """MLP that predicts state increment delta_x given z_t and u_applied.

    Attributes
    ----------
    hidden_dim      : Must match the transformer's hidden_dim.
    action_dim      : Dimensionality of the action space.
    state_dim       : Dimensionality of the full state x_t.
    head_hidden_dim : Width of each hidden layer.
    num_layers      : Number of hidden layers (not counting the output layer).
    activation      : Activation function string ('gelu', 'relu', 'tanh', ...).
    """

    hidden_dim: int
    action_dim: int
    state_dim: int
    head_hidden_dim: int = 128
    num_layers: int = 3
    activation: str = "gelu"

    def setup(self) -> None:
        # Input layer norm — applied to the concatenated [z_t, u_applied]
        self.ln = nn.LayerNorm()

        # Hidden layers: first layer accepts (hidden_dim + action_dim) inputs
        in_dim = self.hidden_dim + self.action_dim
        self.first_layer = nn.Dense(self.head_hidden_dim, kernel_init=default_nn_init())

        # Remaining hidden layers (input width is head_hidden_dim)
        self.hidden_layers = [
            nn.Dense(self.head_hidden_dim, kernel_init=default_nn_init())
            for _ in range(self.num_layers - 1)
        ]

        # Output projection: predict state increment (unbounded)
        self.out_proj = nn.Dense(self.state_dim, kernel_init=default_nn_init())

    def __call__(
        self,
        z_t: jnp.ndarray,          # (batch, hidden_dim)
        u_applied: jnp.ndarray,    # (batch, action_dim)
    ) -> jnp.ndarray:              # (batch, state_dim)
        """Predict the state increment delta_x_hat.

        Parameters
        ----------
        z_t       : Transformer last-token latent. Shape: (batch, hidden_dim).
        u_applied : Full applied action (u_nom + delta_u). Shape: (batch, action_dim).

        Returns
        -------
        delta_x_hat : Predicted state increment. Shape: (batch, state_dim).
                      Caller computes x_hat_next = x_t + delta_x_hat.
        """
        act = get_act_from_str(self.activation)

        # Concatenate latent and action, then normalise
        x = jnp.concatenate([z_t, u_applied], axis=-1)   # (batch, hidden_dim + action_dim)
        x = self.ln(x)

        # Forward through MLP
        x = act(self.first_layer(x))
        for layer in self.hidden_layers:
            x = act(layer(x))

        return self.out_proj(x)

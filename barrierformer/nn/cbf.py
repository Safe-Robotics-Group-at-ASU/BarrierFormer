"""CBF MLP network definition."""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from .utils import default_nn_init, get_act_from_str


class CBFMLP(nn.Module):
    """MLP Control Barrier Function."""

    obs_dim: int
    hidden_sizes: Sequence[int] = (256, 256, 128)
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Compute barrier value h(obs)."""
        act = get_act_from_str(self.activation)

        x = nn.LayerNorm()(obs)

        for size in self.hidden_sizes:
            x = act(nn.Dense(size, kernel_init=default_nn_init())(x))

        h = nn.tanh(nn.Dense(1, kernel_init=default_nn_init())(x))
        return h

"""
CBF MLP network definition.

Maps a full observation vector o_t to a scalar barrier value h(o_t)
bounded in [-1, +1] via a final tanh activation:

    h(o_t) > 0  →  safe region
    h(o_t) < 0  →  unsafe region

This file contains ONLY the neural-network definition.
Horizon evaluation and DCBF violation computation live in algo/cbf_eval.py.
Training losses live in algo/losses.py.
"""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from .utils import default_nn_init, get_act_from_str


class CBFMLP(nn.Module):
    """MLP Control Barrier Function.

    A feedforward network that maps a full observation vector to a scalar
    barrier value in the range (-1, 1).  The output is bounded by a final
    ``tanh`` activation, so that:

        * h(o) > 0  →  the model predicts the state is SAFE
        * h(o) < 0  →  the model predicts the state is UNSAFE

    Architecture
    ------------
    LayerNorm(input)
    → [Dense(hidden_sizes[i]) → activation]  for i in 0..len(hidden_sizes)-1
    → Dense(1) → tanh

    Attributes
    ----------
    obs_dim      : Dimensionality of the input observation vector.
    hidden_sizes : Width of each hidden layer, e.g. ``(256, 256, 128)``.
    activation   : Activation function name ('relu', 'tanh', 'gelu', …).
    """

    obs_dim: int
    hidden_sizes: Sequence[int] = (256, 256, 128)
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Compute barrier value h(obs).

        Parameters
        ----------
        obs : Observation vector.  Shape: ``(batch, obs_dim)``.

        Returns
        -------
        h   : Scalar barrier value.  Shape: ``(batch, 1)``.
              Range: (-1, 1).  Positive → safe, negative → unsafe.
        """
        act = get_act_from_str(self.activation)

        # Normalise input to stabilise training on heterogeneous obs features
        # (state, relative goal, LiDAR distances all live on different scales).
        x = nn.LayerNorm()(obs)

        for size in self.hidden_sizes:
            x = act(nn.Dense(size, kernel_init=default_nn_init())(x))

        # Final projection to scalar, then squeeze into [-1, 1]
        h = nn.tanh(nn.Dense(1, kernel_init=default_nn_init())(x))
        return h  # (batch, 1)

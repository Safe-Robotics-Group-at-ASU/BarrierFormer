"""Discrete-time CBF evaluation over a predicted rollout horizon."""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp


def discrete_cbf_violation(
    h_k: jnp.ndarray,
    h_k1: jnp.ndarray,
    alpha: float,
) -> jnp.ndarray:
    """Per-step DCBF violation (non-negative)."""
    return jax.nn.relu(-(h_k1 + (alpha - 1.0) * h_k))


def eval_cbf_over_horizon(
    cbf_apply_fn: Callable,
    cbf_params: dict,
    obs_seq: jnp.ndarray,
    alpha: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the CBF network over a rollout horizon and compute violations."""

    def apply_single_step(obs_t: jnp.ndarray) -> jnp.ndarray:
        h = cbf_apply_fn(cbf_params, obs_t)
        return h.squeeze(-1)

    h_all = jax.vmap(apply_single_step)(obs_seq.transpose(1, 0, 2))
    h_seq = h_all.transpose(1, 0)

    h_k  = h_seq[:, :-1]
    h_k1 = h_seq[:, 1:]

    violations = discrete_cbf_violation(h_k, h_k1, alpha)

    return h_seq, violations

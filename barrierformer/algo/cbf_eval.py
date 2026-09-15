"""
Discrete-time CBF evaluation over a predicted rollout horizon.

All functions in this module are pure JAX — no class state, no env calls,
no training-loop side effects.  They operate on pre-computed observation
sequences produced by algo/rollout.py.

Discrete-time CBF (DCBF) condition
-----------------------------------
For a barrier function h: obs → ℝ and decay rate α ∈ (0, 1], safety requires:
    h(o_{k+1}) - h(o_k) + α · h(o_k) ≥ 0 ⟺  h(o_{k+1}) ≥ (1 - α) · h(o_k)
A positive violation at step k means the rollout FAILS this condition at k.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp


def discrete_cbf_violation(
    h_k: jnp.ndarray,    # (...) barrier value at step k
    h_k1: jnp.ndarray,   # (...) barrier value at step k+1
    alpha: float,
) -> jnp.ndarray:
    """Per-step DCBF violation (non-negative).

    Computes how much the discrete-time CBF decrease condition is violated:

        condition : h_{k+1} - h_k + α · h_k  ≥  0
        violation : relu( -(h_{k+1} + (α-1)·h_k) ) = relu( -h_{k+1} - (α-1)·h_k )

    Parameters
    ----------
    h_k   : Barrier value at the current rollout step.     Arbitrary shape.
    h_{k+1}  : Barrier value at the next rollout step.     Same shape as h_k.
    alpha : CBF decay rate.  Typical range: (0, 1].

    Returns
    -------
    violation : Non-negative array, same shape as h_k.
                Zero means the condition is satisfied; positive means violated.
    """
    # DCBF condition: h_{k+1} - h_k + alpha * h_k >= 0
    # Rearranged:     h_{k+1} + (alpha - 1) * h_k >= 0
    return jax.nn.relu(-(h_k1 + (alpha - 1.0) * h_k))


def eval_cbf_over_horizon(
    cbf_apply_fn: Callable,        # fn(cbf_params, obs) → (batch, 1)
    cbf_params: dict,
    obs_seq: jnp.ndarray,          # (batch, H, obs_dim)
    alpha: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the CBF network over a rollout horizon and compute violations.

    Applies ``cbf_apply_fn`` at every step along the horizon, then computes
    the DCBF violation between every consecutive pair of steps.

    Parameters
    ----------
    cbf_apply_fn : Callable that evaluates the CBF.
                   Signature: ``(cbf_params, obs: (batch, obs_dim)) → (batch, 1)``.
    cbf_params   : Frozen parameter dict for the CBF network (CBFMLP.init output).
    obs_seq      : Predicted observation sequence.  Shape: ``(batch, H, obs_dim)``.
    alpha        : DCBF decay rate α ∈ (0, 1].

    Returns
    -------
    h_seq      : Barrier values at each horizon step.
                 Shape: ``(batch, H)``.
    violations : DCBF violations between consecutive steps.
                 Shape: ``(batch, H-1)``.
                 ``violations[:, k]`` = violation between steps k and k+1.
    """
    # Evaluate CBF at every horizon step.
    # We vmap over the time axis (axis=1 of obs_seq) to apply cbf_apply_fn
    # to each (batch, obs_dim) slice independently.

    def apply_single_step(obs_t: jnp.ndarray) -> jnp.ndarray:
        # obs_t : (batch, obs_dim)
        h = cbf_apply_fn(cbf_params, obs_t)  # (batch, 1)
        return h.squeeze(-1)                 # (batch,)

    # Transpose obs_seq to (H, batch, obs_dim), vmap over H, transpose back.
    # jax.vmap over the leading axis of the transposed array gives:
    #   in:  (H, batch, obs_dim)  →  out: (H, batch)
    h_all = jax.vmap(apply_single_step)(obs_seq.transpose(1, 0, 2))
    # h_all: (H, batch)
    h_seq = h_all.transpose(1, 0)           # (batch, H)

    # DCBF violations for consecutive step pairs k = 0 .. H-2
    h_k  = h_seq[:, :-1]   # (batch, H-1)  — barrier at step k
    h_k1 = h_seq[:, 1:]    # (batch, H-1)  — barrier at step k+1

    violations = discrete_cbf_violation(h_k, h_k1, alpha)
    # violations: (batch, H-1), all values ≥ 0

    return h_seq, violations

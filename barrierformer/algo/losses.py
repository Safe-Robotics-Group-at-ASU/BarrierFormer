"""Differentiable training losses for the transformer CBF framework."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def cbf_horizon_loss(
    violations: jnp.ndarray,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Differentiable worst-case CBF violation over the rollout horizon."""
    log_sum = jax.scipy.special.logsumexp(
        violations / temperature, axis=-1
    )
    per_batch_loss = temperature * log_sum

    return jnp.mean(per_batch_loss)


def dynamics_prediction_loss(
    delta_x_hat: jnp.ndarray,
    delta_x_true: jnp.ndarray,
) -> jnp.ndarray:
    """MSE loss on the dynamics-head state-increment prediction."""
    return jnp.mean(jnp.sum((delta_x_hat - delta_x_true) ** 2, axis=-1))

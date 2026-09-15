"""
Differentiable training losses for the transformer CBF framework.

These functions are called *only* during batch training — NOT during online
rollout / data collection.  All operations must be fully differentiable
(no hard max, no stop_gradient except where explicitly noted).

Stability note on log-sum-exp
------------------------------
A numerically stable evaluation of

    temperature · log Σ_k exp(v_k / temperature)

subtracts the per-sample maximum before exponentiation and adds it back:

    temperature · ( max_v + log Σ_k exp((v_k - max_v) / temperature) )

This avoids floating-point overflow for large violations and underflow for
near-zero violations.  JAX's ``jax.scipy.special.logsumexp`` does this
automatically when ``axis`` is specified.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def cbf_horizon_loss(
    violations: jnp.ndarray,  # (batch, H-1)  per-step DCBF violations (≥ 0)
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Differentiable worst-case CBF violation over the rollout horizon.
    Replaces a non-differentiable ``max()`` over H-1 violation values with a
    numerically stable *log-sum-exp* approximation:
        loss_i = temperature · log Σ_{k=0}^{H-2} exp( violations_{i,k} / temperature )

    Scalar output = mean over the batch.

    Behaviour at extreme temperatures
    ----------------------------------
    * temperature → 0 : approaches ``max_k violations_{i,k}``
                        (sharpest; most sensitive to worst step)
    * temperature → ∞ : approaches ``temperature · log(H-1)  +  mean violations_i``
                        (softest; similar to mean reduction)
    * temperature = 1 : reasonable default for training

    Parameters
    ----------
    violations  : Per-step DCBF violations produced by
                  ``algo.cbf_eval.eval_cbf_over_horizon``.
                  Shape: ``(batch, H-1)``.  All values are ≥ 0.
    temperature : Log-sum-exp smoothing temperature (> 0).

    Returns
    -------
    loss : Scalar differentiable safety penalty (mean over batch).
    """
    # Numerically stable log-sum-exp over the horizon axis.
    # jax.scipy.special.logsumexp(x, axis) computes log Σ exp(x_i) stably.
    # We scale the input by 1/temperature and the output by temperature.
    log_sum = jax.scipy.special.logsumexp(
        violations / temperature, axis=-1
    )                                      # (batch,)
    per_batch_loss = temperature * log_sum  # (batch,)

    return jnp.mean(per_batch_loss)         # scalar


def dynamics_prediction_loss(
    delta_x_hat: jnp.ndarray,   # (batch, state_dim)
    delta_x_true: jnp.ndarray,  # (batch, state_dim)
) -> jnp.ndarray:
    """MSE loss on the dynamics-head state-increment prediction.

    Penalises inaccurate one-step state predictions during training.
    Used alongside the CBF safety loss in the total training objective.

    Parameters
    ----------
    delta_x_hat  : Predicted state increments from DynamicsHead.
                   Shape: ``(batch, state_dim)``.
    delta_x_true : Ground-truth state increments (x_{t+1} - x_t from env).
                   Shape: ``(batch, state_dim)``.

    Returns
    -------
    loss : Scalar mean squared error averaged over batch and state dimensions.
    """
    return jnp.mean(jnp.sum((delta_x_hat - delta_x_true) ** 2, axis=-1))

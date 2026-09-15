"""
Autoregressive rollout logic for the transformer CBF framework.

Responsibilities of this module (algo layer)
--------------------------------------------
* Extract the transformer context window from the current environment graph.
* Build observation-reconstruction and nominal-action closures from frozen
  graph data (goals, obstacles).  The actual LiDAR computation is delegated
  to env/utils.py — nothing is re-implemented here.
* Run the H-step autoregressive rollout by calling the CausalTransformerPolicy
  forward pass repeatedly and sliding the context window.

What this module does NOT do
-----------------------------
* Define neural network architectures      → nn/ layer
* Compute CBF violations or safety losses  → algo/cbf_eval.py, algo/losses.py
* Execute against the real simulator       → algo/receding_horizon.py
* Modify environment state                 → env/ layer

Context-window convention (matches CausalTransformerPolicy)
-------------------------------------------------------------
env stores:
    obs_history    : (history_len,     n_agents, obs_dim)   where T = history_len - 1
    action_history : (history_len - 1, n_agents, act_dim)

policy expects:
    obs_seq    : (batch, T+1, obs_dim)  = (n_agents, history_len, obs_dim)
    action_seq : (batch, T,   act_dim)  = (n_agents, history_len-1, act_dim)
    u_nom      : (batch, act_dim)       — nominal action for the current step

The two shapes are consistent with T = history_len - 1.

Nominal-action recomputation during rollout
--------------------------------------------
At each autoregressive step k the nominal action is recomputed from the
*predicted* state x_hat_k using the LQR law frozen at the start of the
rollout.  This mirrors how DoubleIntegrator.u_ref works but operates on
raw arrays (no graph required) so it can run inside a Python loop that
JAX traces and unrolls.
"""

from __future__ import annotations

import functools as ft
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from barrierformer.nn.model import RolloutOutput
from barrierformer.utils.graph import GraphsTuple


# ─────────────────────────────────────────────────────────────────────────────
# Context extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_context(
    graph: GraphsTuple,
    n_agents: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract the transformer context window and current state from a graph.

    Reads obs_history and action_history stored in
    graph.env_states and reshapes them to the format expected by CausalTransformerPolicy:
        obs_seq    : (n_agents, history_len,     obs_dim)  ← (batch, T+1, obs_dim)
        action_seq : (n_agents, history_len - 1, act_dim)  ← (batch, T,   act_dim)
    where T = history_len - 1.

    Parameters
    ----------
    graph    : Current environment graph (DoubleIntegrator.EnvGraphsTuple).
    n_agents : Number of agents — needed for ``graph.type_states``.

    Returns
    -------
    obs_seq    : Shape (n_agents, history_len, obs_dim).
    action_seq : Shape (n_agents, history_len - 1, act_dim).
    x_current  : Current agent state.  Shape (n_agents, state_dim).
    """
    # obs_history    : (history_len,     n_agents, obs_dim)
    # action_history : (history_len - 1, n_agents, act_dim)
    obs_history    = graph.env_states.obs_history
    action_history = graph.env_states.action_history

    # Transpose time and agent axes so that batch (agent) is axis 0.
    obs_seq    = obs_history.transpose(1, 0, 2)     # (n_agents, history_len,   obs_dim)
    action_seq = action_history.transpose(1, 0, 2)  # (n_agents, history_len-1, act_dim)

    # Current agent state — type_idx=0 is the AGENT node type.
    x_current = graph.type_states(type_idx=0, n_type=n_agents)  # (n_agents, state_dim)

    return obs_seq, action_seq, x_current


# ─────────────────────────────────────────────────────────────────────────────
# Environment-callback factories (freeze env data into pure closures)
# ─────────────────────────────────────────────────────────────────────────────

def make_obs_from_state_fn(
    goal_states: jnp.ndarray,  # (n_agents, state_dim)  — frozen from graph at time t
    obstacles,                 # Obstacle pytree         — frozen from graph at time t
    n_rays: int,               # number of LiDAR beams  (static: n_rays == max_returns)
    comm_radius: float,        # LiDAR sensing range
    num_agents: int,
    env=None,                  # if given, delegate to env._get_obs (3D-safe, matches training)
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build an observation-reconstruction closure for predicted agent states.

    Delegates LiDAR raycasting to env.utils.get_lidar (env/ layer) and computes the relative goal vector inline (trivial arithmetic). 
    Nothing from this function is re-implemented relative to the env : it is purely a convenience adapter.

    Parameters
    ----------
    goal_states : Goal positions/states frozen at the *start* of the rollout.
                  Shape: (n_agents, state_dim).
    obstacles   : Obstacle pytree (Rectangle for DoubleIntegrator).
                  Frozen from the real graph : unchanged during the rollout.
    n_rays      : Number of LiDAR rays (== PARAMS["n_rays"]).
    comm_radius : Agent sensing range (== PARAMS["comm_radius"]).
    num_agents  : Number of agents (needed for reshape after LiDAR flat).

    Returns
    -------
    obs_from_state : Callable ``(n_agents, state_dim) → (n_agents, obs_dim)``
                     that can be called inside a JAX-traced rollout loop.
    """
    # Preferred path: delegate to the env's own observation builder so the
    # reconstructed obs matches training *exactly* and supports 3D envs
    # (CrazyFlie: state 12 + rel_goal 3 + n_rays*5). The inline 2D Way-2
    # implementation below is kept as a fallback for callers that pass env=None.
    if env is not None:
        def obs_from_state_env(agent_states: jnp.ndarray) -> jnp.ndarray:
            return env._get_obs(agent_states, goal_states, obstacles)
        return obs_from_state_env

    # Import from env layer — computation lives there.
    import numpy as _np
    from barrierformer.env.utils import get_lidar
    from barrierformer.utils.utils import jax_vmap

    _get_lidar_vmapped = jax_vmap(
        ft.partial(
            get_lidar,
            obstacles=obstacles,
            num_beams=n_rays,
            sense_range=comm_radius,
            max_returns=n_rays,
        )
    )

    # Precompute fixed ray angles (Way-2 encoding, matches DoubleIntegrator._get_obs).
    _thetas  = _np.linspace(-_np.pi, _np.pi - 2 * _np.pi / n_rays, n_rays)
    _ray_cos = jnp.array(_np.cos(_thetas))   # (n_rays,)
    _ray_sin = jnp.array(_np.sin(_thetas))   # (n_rays,)

    def obs_from_state(agent_states: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct 134-dim Way-2 observation from predicted agent state.

        Mirrors DoubleIntegrator._get_obs (Way-2):
            o_t = [ x_t(4) | r_t^goal(2) | (hit_mask, dist_norm, cos_θ, sin_θ)×32 ]
        """
        lidar_xy = _get_lidar_vmapped(agent_states[:, :2])   # (n_agents, n_rays, 2)

        def encode_agent(agent_pos, lidar_hits):
            # stop_gradient on lidar_hits: raytracing det=0 singularities cause
            # NaN gradients. SDF below provides the clean differentiable signal.
            hits_sg  = jax.lax.stop_gradient(lidar_hits)
            rel      = hits_sg - agent_pos[None, :]
            dist     = jnp.sqrt(jnp.sum(rel ** 2, axis=-1) + 1e-8)
            hit_mask = jnp.less(dist, comm_radius).astype(jnp.float32)

            # Signed distance to nearest obstacle: mirrors DoubleIntegrator._get_obs.
            all_sdf  = jax.vmap(lambda obs_j: obs_j.signed_distance(agent_pos))(obstacles)
            min_sdf  = all_sdf.min()
            is_inside = min_sdf < 0.0

            dist_norm = jnp.where(
                is_inside,
                jnp.clip(min_sdf / comm_radius, -1.0, 0.0),
                jnp.clip(jnp.where(hit_mask > 0.5, dist / comm_radius, 1.0), 0.0, 1.0),
            )
            encoded  = jnp.stack(
                [hit_mask, dist_norm, _ray_cos, _ray_sin], axis=-1)  # (n_rays, 4)
            return encoded.reshape(-1)                                 # (n_rays*4,)

        lidar_encoded = jax.vmap(encode_agent)(
            agent_states[:, :2], lidar_xy)                           # (n_agents, n_rays*4)
        rel_goal = goal_states[:, :2] - agent_states[:, :2]         # (n_agents, 2)

        return jnp.concatenate([agent_states, rel_goal, lidar_encoded], axis=-1)

    return obs_from_state


def make_u_nom_fn(
    env,
    goal_states: jnp.ndarray,   # (n_agents, state_dim) — frozen from graph at time t
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a nominal-action closure that delegates to env.u_nom_from_state.

    Parameters
    ----------
    env         : Environment instance (must implement u_nom_from_state).
    goal_states : Goal states frozen at the start of the rollout.
                  Shape: ``(n_agents, state_dim)``.

    Returns
    -------
    u_nom_fn : Callable ``(n_agents, state_dim) → (n_agents, action_dim)``.
    """

    def u_nom_fn(agent_states: jnp.ndarray) -> jnp.ndarray:
        return env.u_nom_from_state(agent_states, goal_states)

    return u_nom_fn


# ─────────────────────────────────────────────────────────────────────────────
# Autoregressive rollout
# ─────────────────────────────────────────────────────────────────────────────

def autoregressive_rollout(
    policy_apply_fn: Callable,
    policy_params: dict,
    obs_seq: jnp.ndarray,               # (n_agents, T+1, obs_dim)
    action_seq: jnp.ndarray,            # (n_agents, T,   act_dim)
    x_current: jnp.ndarray,             # (n_agents, state_dim)
    horizon: int,
    obs_from_state_fn: Callable,        # (n_agents, state_dim) → (n_agents, obs_dim)
    u_nom_from_state_fn: Callable,      # (n_agents, state_dim) → (n_agents, act_dim)
    action_lb: jnp.ndarray,             # (act_dim,) lower clip bound — matches env.clip_action
    action_ub: jnp.ndarray,             # (act_dim,) upper clip bound — matches env.clip_action
    deterministic: bool = True,
) -> RolloutOutput:
    """H-step autoregressive predictive rollout using the transformer policy.

    At each step k (k = 0 .. H-1):

      1. Recompute the nominal action from the current predicted state:
             u_nom_k = u_nom_from_state_fn(x_hat_k)
      2. Run the policy single-step forward pass:
             (delta_u, u_applied, delta_x_hat) = policy(obs_seq, action_seq, u_nom_k)
      3. Integrate the predicted state:
             x_hat_{k+1} = x_hat_k + delta_x_hat
      4. Reconstruct the observation for the predicted next state:
             o_hat_{k+1} = obs_from_state_fn(x_hat_{k+1})
      5. Slide the context window by one step:
             obs_seq    ← concat(obs_seq[:, 1:, :],    o_hat_{k+1}[:, None, :], axis=1)
             action_seq ← concat(action_seq[:, 1:, :], u_applied[:, None, :],   axis=1)

    This is a plain Python for-loop so JAX will unroll the loop at trace time
    when the function is JIT-compiled at the call site.  For large horizons,
    consider wrapping in ``jax.lax.scan`` if compilation time is a concern.

    Parameters
    ----------
    policy_apply_fn      : ``model.apply`` — Flax module apply function.
                           Called as ``policy_apply_fn(params, obs_seq,
                           action_seq, u_nom, deterministic)``.
    policy_params        : Frozen parameter dict from ``model.init``.
    obs_seq              : Initial observation context window.
                           Shape: ``(n_agents, T+1, obs_dim)``.
    action_seq           : Initial action context window.
                           Shape: ``(n_agents, T, act_dim)``.
    x_current            : Initial agent state (the real state at time t).
                           Shape: ``(n_agents, state_dim)``.
    horizon              : Number of rollout steps H.
    obs_from_state_fn    : Observation reconstruction closure.
                           Build with ``make_obs_from_state_fn``.
    u_nom_from_state_fn  : Nominal action closure (LQR from predicted state).
                           Build with ``make_u_nom_fn``.
    action_lb            : Action lower bound used by ``env.clip_action``.
                           Applied to ``u_applied`` before sliding it into
                           ``action_seq`` so that the rollout context exactly
                           mirrors what ``env.step()`` stores in
                           ``graph.env_states.action_history``.
    action_ub            : Action upper bound (same rationale as ``action_lb``).
    deterministic        : Passed to the policy to disable dropout.

    Returns
    -------
    RolloutOutput with:
        x_hat_seq : ``(n_agents, H, state_dim)``  — predicted states
        o_hat_seq : ``(n_agents, H, obs_dim)``    — predicted observations
        u_seq     : ``(n_agents, H, act_dim)``    — applied actions
    """
    x_hat_list: list = []
    o_hat_list: list = []
    u_list:     list = []

    x_hat = x_current   # running state estimate: (n_agents, state_dim)

    for _ in range(horizon):
        # ── 1. Recompute nominal action at current imagined state ─────────────
        # k=0: x_hat == x_current so matches the SQP label reference exactly.
        # k>0: fresh local LQR so the transformer only learns safety corrections
        #      relative to a locally-correct baseline at each horizon step.
        u_nom = u_nom_from_state_fn(x_hat)             # (n_agents, act_dim)

        # ── 2. Policy single-step forward pass ───────────────────────────────
        # Signature: policy_apply_fn(params, obs_seq, action_seq, u_nom, det)
        # Returns:   ModelOutput(z_t, delta_u, u_applied, delta_x_hat, …)
        out = policy_apply_fn(
            policy_params,
            obs_seq,
            action_seq,
            u_nom,
            deterministic,
        )

        # ── 3. Integrate predicted state ─────────────────────────────────────
        x_hat = x_hat + out.delta_x_hat                # (n_agents, state_dim)

        # ── 4. Reconstruct observation — calls env layer ──────────────────────
        o_hat = obs_from_state_fn(x_hat)               # (n_agents, obs_dim)

        x_hat_list.append(x_hat)
        o_hat_list.append(o_hat)
        u_list.append(out.u_applied)

        # ── 5. Slide context window forward by one step ───────────────────────
        obs_seq = jnp.concatenate(
            [obs_seq[:, 1:, :], o_hat[:, None, :]], axis=1
        )
        # Clip u_applied before sliding into action_seq.
        # env.step() stores clip(action) in action_history, so the rollout
        # context must use the same clipped value to avoid a training/inference
        # distribution mismatch in the action channel.
        u_applied_clipped = jnp.clip(out.u_applied, action_lb, action_ub)
        action_seq = jnp.concatenate(
            [action_seq[:, 1:, :], u_applied_clipped[:, None, :]], axis=1
        )

    # Collect: list of H arrays → (n_agents, H, dim)
    x_hat_seq = jnp.stack(x_hat_list, axis=1)   # (n_agents, H, state_dim)
    o_hat_seq = jnp.stack(o_hat_list, axis=1)   # (n_agents, H, obs_dim)
    u_seq     = jnp.stack(u_list,     axis=1)   # (n_agents, H, act_dim)

    return RolloutOutput(x_hat_seq=x_hat_seq, o_hat_seq=o_hat_seq, u_seq=u_seq)

"""Autoregressive rollout logic for the transformer CBF framework."""

from __future__ import annotations

import functools as ft
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from barrierformer.nn.model import RolloutOutput
from barrierformer.utils.graph import GraphsTuple


def extract_context(
    graph: GraphsTuple,
    n_agents: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract the transformer context window and current state from a graph."""
    obs_history    = graph.env_states.obs_history
    action_history = graph.env_states.action_history

    obs_seq    = obs_history.transpose(1, 0, 2)
    action_seq = action_history.transpose(1, 0, 2)

    x_current = graph.type_states(type_idx=0, n_type=n_agents)

    return obs_seq, action_seq, x_current


def make_obs_from_state_fn(
    goal_states: jnp.ndarray,
    obstacles,
    n_rays: int,
    comm_radius: float,
    num_agents: int,
    ray_order: bool = False,
    state_obs: str = "raw",
    env=None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build an observation-reconstruction closure for predicted agent states."""
    if env is not None and getattr(env, "USE_ENV_OBS_IN_ROLLOUT", False):
        def obs_from_env(agent_states: jnp.ndarray) -> jnp.ndarray:
            return env._get_obs(agent_states, goal_states, obstacles)
        return obs_from_env

    import numpy as _np
    from barrierformer.env.utils import get_lidar, encode_state_obs
    from barrierformer.utils.utils import jax_vmap

    _get_lidar_vmapped = jax_vmap(
        ft.partial(
            get_lidar,
            obstacles=obstacles,
            num_beams=n_rays,
            sense_range=comm_radius,
            max_returns=n_rays,
            sort=not ray_order,
        )
    )

    _thetas  = _np.linspace(-_np.pi, _np.pi - 2 * _np.pi / n_rays, n_rays)
    _ray_cos = jnp.array(_np.cos(_thetas))
    _ray_sin = jnp.array(_np.sin(_thetas))

    def obs_from_state(agent_states: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct the observation from a predicted agent state."""
        lidar_xy = _get_lidar_vmapped(agent_states[:, :2])

        def encode_agent(agent_pos, lidar_hits):
            hits_sg  = jax.lax.stop_gradient(lidar_hits)
            rel      = hits_sg - agent_pos[None, :]
            dist     = jnp.sqrt(jnp.sum(rel ** 2, axis=-1) + 1e-8)
            hit_mask = jnp.less(dist, comm_radius).astype(jnp.float32)

            all_sdf  = jax.vmap(lambda obs_j: obs_j.signed_distance(agent_pos))(obstacles)
            min_sdf  = all_sdf.min()
            is_inside = min_sdf < 0.0

            dist_norm = jnp.where(
                is_inside,
                jnp.clip(min_sdf / comm_radius, -1.0, 0.0),
                jnp.clip(jnp.where(hit_mask > 0.5, dist / comm_radius, 1.0), 0.0, 1.0),
            )
            encoded  = jnp.stack(
                [hit_mask, dist_norm, _ray_cos, _ray_sin], axis=-1)
            return encoded.reshape(-1)

        lidar_encoded = jax.vmap(encode_agent)(
            agent_states[:, :2], lidar_xy)
        rel_goal = goal_states[:, :2] - agent_states[:, :2]

        return jnp.concatenate([encode_state_obs(agent_states, state_obs), rel_goal, lidar_encoded], axis=-1)

    return obs_from_state


def make_u_nom_fn(
    env,
    goal_states: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a nominal-action closure that delegates to env.u_nom_from_state."""

    def u_nom_fn(agent_states: jnp.ndarray) -> jnp.ndarray:
        return env.u_nom_from_state(agent_states, goal_states)

    return u_nom_fn


def autoregressive_rollout(
    policy_apply_fn: Callable,
    policy_params: dict,
    obs_seq: jnp.ndarray,
    action_seq: jnp.ndarray,
    x_current: jnp.ndarray,
    horizon: int,
    obs_from_state_fn: Callable,
    u_nom_from_state_fn: Callable,
    action_lb: jnp.ndarray,
    action_ub: jnp.ndarray,
    state_lb: jnp.ndarray = None,
    state_ub: jnp.ndarray = None,
    deterministic: bool = True,
) -> RolloutOutput:
    """H-step autoregressive predictive rollout using the transformer policy."""
    x_hat_list: list = []
    o_hat_list: list = []
    u_list:     list = []

    x_hat = x_current

    for _ in range(horizon):
        u_nom = u_nom_from_state_fn(x_hat)

        out = policy_apply_fn(
            policy_params,
            obs_seq,
            action_seq,
            u_nom,
            deterministic,
        )

        x_hat = x_hat + out.delta_x_hat
        if state_lb is not None:
            x_hat = jnp.clip(x_hat, state_lb, state_ub)

        o_hat = obs_from_state_fn(x_hat)

        x_hat_list.append(x_hat)
        o_hat_list.append(o_hat)
        u_list.append(out.u_applied)

        obs_seq = jnp.concatenate(
            [obs_seq[:, 1:, :], o_hat[:, None, :]], axis=1
        )
        u_applied_clipped = jnp.clip(out.u_applied, action_lb, action_ub)
        action_seq = jnp.concatenate(
            [action_seq[:, 1:, :], u_applied_clipped[:, None, :]], axis=1
        )

    x_hat_seq = jnp.stack(x_hat_list, axis=1)
    o_hat_seq = jnp.stack(o_hat_list, axis=1)
    u_seq     = jnp.stack(u_list,     axis=1)

    return RolloutOutput(x_hat_seq=x_hat_seq, o_hat_seq=o_hat_seq, u_seq=u_seq)

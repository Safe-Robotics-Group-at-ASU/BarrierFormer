"""Receding-horizon (MPC-style) action execution for online data collection."""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp

from barrierformer.nn.model import RolloutOutput
from barrierformer.utils.graph import GraphsTuple
from barrierformer.algo.rollout import (
    extract_context,
    make_obs_from_state_fn,
    make_u_nom_fn,
    autoregressive_rollout,
)


def receding_horizon_act(
    policy_apply_fn,
    policy_params: dict,
    graph: GraphsTuple,
    env,
    n_agents: int,
    horizon: int,
    deterministic: bool = True,
    return_full_rollout: bool = False,
) -> Tuple[jnp.ndarray, Optional[RolloutOutput]]:
    """Execute one receding-horizon step: plan H steps, return the first action."""
    comm_radius = env._params["comm_radius"]
    n_rays      = env._params["n_rays"]
    action_lb, action_ub = env.action_lim()
    state_lb, state_ub   = env.state_lim()

    obs_seq, action_seq, x_current = extract_context(graph, n_agents)

    goal_states = graph.env_states.goal
    obstacles   = graph.env_states.obstacle

    obs_from_state_fn = make_obs_from_state_fn(
        goal_states=goal_states,
        obstacles=obstacles,
        n_rays=n_rays,
        comm_radius=comm_radius,
        num_agents=n_agents,
        ray_order=env._params.get("lidar_ray_order", False),
        state_obs=env._params.get("state_obs", "raw"),
        env=env,
    )
    u_nom_from_state_fn = make_u_nom_fn(env=env, goal_states=goal_states)

    rollout_out: RolloutOutput = autoregressive_rollout(
        policy_apply_fn=policy_apply_fn,
        policy_params=policy_params,
        obs_seq=obs_seq,
        action_seq=action_seq,
        x_current=x_current,
        horizon=horizon,
        obs_from_state_fn=obs_from_state_fn,
        u_nom_from_state_fn=u_nom_from_state_fn,
        action_lb=action_lb,
        action_ub=action_ub,
        state_lb=state_lb,
        state_ub=state_ub,
        deterministic=deterministic,
    )

    u_first = rollout_out.u_seq[:, 0, :]

    return u_first, (rollout_out if return_full_rollout else None)

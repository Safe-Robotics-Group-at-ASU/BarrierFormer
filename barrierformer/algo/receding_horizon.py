"""
Receding-horizon (MPC-style) action execution for online data collection.

This module is the entry point for real-time interaction with the environment.
It implements the receding-horizon principle:

    Plan H steps ahead using the transformer policy → act for 1 step → replan.

Protocol for one real timestep t
----------------------------------
1. Extract the current context window from the real graph (obs_history,
   action_history stored in graph.env_states — populated by the env step).
2. Freeze goal positions and obstacle geometry from the real graph.
3. Run an H-step autoregressive predictive rollout (algo/rollout.py).
4. Return ONLY the first predicted action ``u_seq[:, 0, :]``.
5. The caller steps the *real* simulator with that action, appends the
   true transition to the history buffer, and calls this function again.

The predicted rollout (steps 2..H) is discarded — it is never applied to the
real environment.  This is identical to how receding-horizon MPC works.

Note: for training-time safety loss computation (not real-time execution),
use ``algo/rollout.py`` + ``algo/cbf_eval.py`` + ``algo/losses.py`` directly.
"""

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
    policy_apply_fn,                # model.apply — Flax module apply function
    policy_params: dict,            # CausalTransformerPolicy frozen params
    graph: GraphsTuple,             # current real environment graph
    env,                            # environment instance (provides u_nom_from_state)
    n_agents: int,
    horizon: int,
    # ── Options ─────────────────────────────────────────────────────────────
    deterministic: bool = True,
    return_full_rollout: bool = False,
) -> Tuple[jnp.ndarray, Optional[RolloutOutput]]:
    """Execute one receding-horizon step: plan H steps, return the first action.

    This is the single function to call during data collection / environment
    interaction.  It is NOT used for training loss computation.

    Parameters
    ----------
    policy_apply_fn     : ``model.apply`` from a ``CausalTransformerPolicy``.
    policy_params       : Frozen parameter dict (output of ``model.init``).
    graph               : Current real environment graph.  Must carry
                          ``env_states.obs_history``, ``env_states.action_history``,
                          ``env_states.goal``, ``env_states.obstacle``.
    env                 : Environment instance.  Must implement ``u_nom_from_state``,
                          ``action_lim()``, and expose ``_params["comm_radius"]`` and
                          ``_params["n_rays"]``.
    n_agents            : Number of agents.
    horizon             : Predictive rollout horizon H.
    deterministic       : If True, disables transformer dropout.
    return_full_rollout : If True, also return the full ``RolloutOutput``
                          (useful for safety monitoring or debugging).
                          Default False to avoid unnecessary computation.

    Returns
    -------
    u_first : First predicted action to apply to the real simulator.
              Shape: ``(n_agents, action_dim)``.
    rollout : Full ``RolloutOutput`` if ``return_full_rollout=True``, else None.
              Shapes: x_hat_seq ``(n_agents, H, state_dim)``,
                      o_hat_seq ``(n_agents, H, obs_dim)``,
                      u_seq     ``(n_agents, H, act_dim)``.
    """
    comm_radius = env._params["comm_radius"]
    n_rays      = env._params["n_rays"]
    action_lb, action_ub = env.action_lim()

    # ── 1. Extract context window from the current real graph ─────────────
    obs_seq, action_seq, x_current = extract_context(graph, n_agents)

    # ── 2. Freeze goal and obstacle data from the real graph ──────────────
    goal_states = graph.env_states.goal      # (n_agents, state_dim)
    obstacles   = graph.env_states.obstacle  # Obstacle pytree (Rectangle for DI)

    # ── 3. Build environment callbacks (pure closures over frozen data) ───
    obs_from_state_fn = make_obs_from_state_fn(
        goal_states=goal_states,
        obstacles=obstacles,
        n_rays=n_rays,
        comm_radius=comm_radius,
        num_agents=n_agents,
        env=env,   # delegate to env._get_obs — 3D-safe (CrazyFlie) and matches training
    )
    u_nom_from_state_fn = make_u_nom_fn(env=env, goal_states=goal_states)

    # ── 4. H-step autoregressive predictive rollout ───────────────────────
    rollout_out: RolloutOutput = autoregressive_rollout(
        policy_apply_fn=policy_apply_fn,
        policy_params=policy_params,
        obs_seq=obs_seq,
        action_seq=action_seq,
        x_current=x_current,
        horizon=horizon,
        obs_from_state_fn=obs_from_state_fn,
        u_nom_from_state_fn=u_nom_from_state_fn,
        action_lb=action_lb,   # pass through so rollout clips context actions
        action_ub=action_ub,   # to match what env.step() stores in action_history
        deterministic=deterministic,
    )

    # ── 5. Receding horizon: take only the first predicted action ─────────
    # Remaining horizon steps are discarded.  The real simulator is stepped
    # externally with u_first; the true transition is added to the buffer.
    u_first = rollout_out.u_seq[:, 0, :]    # (n_agents, action_dim)

    return u_first, (rollout_out if return_full_rollout else None)

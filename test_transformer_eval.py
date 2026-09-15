"""
test_transformer_eval.py
========================
Generic evaluator for a saved CausalTransformerPolicy (BarrierFormer) checkpoint.
Supports all environments via --env-id.

Usage:
    python test_transformer_eval.py --env-id DubinsCar --path ./logs/DubinsCar/gcbf_transformer/seed0_XXXXXXXX
    python test_transformer_eval.py --env-id DoubleIntegrator --path ./logs/.../seed0_... --step 100 --n-env 32
    python test_transformer_eval.py --env-id DubinsCar --path ./logs/.../seed0_... --seeds 0 1 2 --n-env 32 --parallel --log

The script reads config.yaml written by train_transformer.py to reconstruct
the exact BarrierFormer / environment configuration used during training.

Supported --env-id values: SingleIntegrator, DoubleIntegrator, LinearDrone, DubinsCar, CrazyFlie
"""

import argparse
import datetime
import os
import pathlib
import pickle
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

from barrierformer.env import ENV
from barrierformer.env.utils import inside_obstacles
from barrierformer.algo.barrierformer import BarrierFormer
from barrierformer.trainer.data import Rollout
from barrierformer.algo.receding_horizon import receding_horizon_act
from barrierformer.utils.utils import jax_vmap

# Environments whose constructor accepts history_len
_ENVS_WITH_HISTORY  = {"DubinsCar", "DoubleIntegrator", "CrazyFlie"}
# Environments whose PARAMS may include use_alt_u_ref
_ENVS_WITH_ALT_UREF = {"DubinsCar"}
# Environments whose PARAMS may include obs_len_range
_ENVS_WITH_OBS_LEN  = {"CrazyFlie"}

# Per-env evaluation defaults — used only when neither --cli nor config.yaml provide a value
_ENV_DEFAULTS = {
    "CrazyFlie": dict(area_size=3.0, n_obs=27, n_rays=32, obs_len_range=[0.1, 0.6]),
}


# ── Environment factory ───────────────────────────────────────────────────────

def _make_env(env_id, num_agents, area_size, max_step, history_len, n_obs, n_rays,
              use_alt_u_ref, obs_len_range=None):
    EnvClass  = ENV[env_id]
    env_params = dict(EnvClass.PARAMS)
    if n_obs is not None:
        env_params["n_obs"] = n_obs
    if n_rays is not None:
        env_params["n_rays"] = n_rays
    if env_id in _ENVS_WITH_ALT_UREF:
        env_params["use_alt_u_ref"] = use_alt_u_ref
    if env_id in _ENVS_WITH_OBS_LEN and obs_len_range is not None:
        env_params["obs_len_range"] = obs_len_range

    kwargs = dict(num_agents=num_agents, area_size=area_size, max_step=max_step, params=env_params)
    if env_id in _ENVS_WITH_HISTORY:
        kwargs["history_len"] = history_len
    return EnvClass(**kwargs)


# ── Statistics helpers ────────────────────────────────────────────────────────

def _stats_within_seed(unsafe_arr, finish_arr):
    """Mean ± std for a single seed's episodes.

    unsafe_arr, finish_arr : (n_epi, n_agents) bool arrays.
    Std is computed across episodes (each episode averaged over agents first).
    """
    safe_epi  = (1 - unsafe_arr).mean(axis=-1)               # (n_epi,)
    fin_epi   = finish_arr.mean(axis=-1)                       # (n_epi,)
    succ_epi  = ((1 - unsafe_arr) * finish_arr).mean(axis=-1) # (n_epi,)
    return (
        float(safe_epi.mean()),  float(safe_epi.std()),
        float(fin_epi.mean()),   float(fin_epi.std()),
        float(succ_epi.mean()),  float(succ_epi.std()),
    )


def _stats_cross_seed(unsafe_arr, finish_arr, epi_per_seed, n_seeds):
    """Mean ± std where std is across seeds (correct aggregate metric).

    Each seed contributes one mean value; the reported std is the std of those
    per-seed means.  Also returns per-seed rate arrays for logging.
    """
    safe_ps, fin_ps, succ_ps = [], [], []
    for i in range(n_seeds):
        sl = slice(i * epi_per_seed, (i + 1) * epi_per_seed)
        u  = unsafe_arr[sl]
        f  = finish_arr[sl]
        safe_ps.append(float((1 - u).mean()))
        fin_ps.append(float(f.mean()))
        succ_ps.append(float(((1 - u) * f).mean()))

    s  = np.array(safe_ps)
    fn = np.array(fin_ps)
    su = np.array(succ_ps)
    return (
        float(s.mean()),  float(s.std()),
        float(fn.mean()), float(fn.std()),
        float(su.mean()), float(su.std()),
        safe_ps, fin_ps, succ_ps,
    )


# ── CSV log helper ────────────────────────────────────────────────────────────

_LOG_HEADER = (
    "timestamp,env_id,ckpt_step,n_agents,n_obs,n_rays,area_size,max_step,"
    "row_type,seed,epi_count,"
    "finish_pct,safe_pct,success_pct,"
    "finish_std_pct,safe_std_pct,success_std_pct\n"
)


def _append_log(log_path, timestamp, env_id, step, n_agents, n_obs, n_rays,
                area_size, max_step, row_type, seed_label, epi_count,
                finish, safe, succ, finish_std=0.0, safe_std=0.0, succ_std=0.0):
    write_header = not os.path.exists(log_path)
    with open(log_path, "a") as f:
        if write_header:
            f.write(_LOG_HEADER)
        f.write(
            f"{timestamp},{env_id},{step},{n_agents},{n_obs},{n_rays},"
            f"{area_size},{max_step},"
            f"{row_type},{seed_label},{epi_count},"
            f"{finish*100:.3f},{safe*100:.3f},{succ*100:.3f},"
            f"{finish_std*100:.3f},{safe_std*100:.3f},{succ_std*100:.3f}\n"
        )


# ── Graph / rollout helpers (used by progressive-removal scripts) ─────────────

def _graph_with_obstacles(env, dense_graph, new_obstacles):
    """Swap the obstacle field in a graph and refresh obs_history."""
    n_agents = env.num_agents
    states = dense_graph.type_states(type_idx=0, n_type=n_agents)
    goals  = dense_graph.env_states.goal
    initial_obs = env._get_obs(states, goals, new_obstacles)
    obs_history = jnp.repeat(initial_obs[None, ...], env.history_len, axis=0)
    action_history = jnp.zeros(
        (env.history_len - 1, n_agents, env.action_dim),
        dtype=dense_graph.env_states.action_history.dtype,
    )
    env_states = env.EnvState(
        agent=states, goal=goals, obstacle=new_obstacles,
        obs_history=obs_history, action_history=action_history,
        step_count=jnp.array(0, dtype=jnp.int32),
    )
    return env.get_graph(env_states)


def rollout_from_graph(env, policy_apply_fn, policy_params, n_agents, horizon,
                       init_graph, key, action_scale=1.0):
    """env.step()-based rollout starting from a caller-supplied init_graph."""
    action_lb, action_ub = env.action_lim()

    def body(graph, _):
        u_first, _ = receding_horizon_act(
            policy_apply_fn=policy_apply_fn,
            policy_params=policy_params,
            graph=graph, env=env, n_agents=n_agents, horizon=horizon,
            deterministic=True, return_full_rollout=False,
        )
        x      = graph.type_states(type_idx=0, n_type=n_agents)
        goals  = graph.env_states.goal
        u_nom  = env.u_nom_from_state(x, goals)
        action = jnp.clip(u_nom + action_scale * (u_first - u_nom), action_lb, action_ub)
        next_graph, reward, cost, done, _ = env.step(graph, action)
        return next_graph, (graph, action, reward, cost, done, next_graph)

    import jax.random as jr
    keys = jr.split(key, env.max_episode_steps)
    _, (graphs, actions, rewards, costs, dones, next_graphs) = jax.lax.scan(
        body, init_graph, keys, length=env.max_episode_steps
    )
    return Rollout(graphs, actions, rewards, costs, dones, next_graphs)


# ── Rollout functions ─────────────────────────────────────────────────────────

def rollout_dyn_head(env, policy_apply_fn, policy_params, n_agents, key, action_scale=1.0):
    """Episode rollout using the transformer dynamics head as the world model.

    x_{t+1} = x_t + delta_x_hat  — no env.step() call.
    Only valid for envs that support history_len (DubinsCar, DoubleIntegrator).
    """
    key_x0, _ = jax.random.split(key)
    graph = env.reset(key_x0)

    x         = graph.type_states(type_idx=0, n_type=n_agents)
    goals     = graph.env_states.goal
    obstacles = graph.env_states.obstacle
    car_r     = env._params["car_radius"]
    action_lb, action_ub = env.action_lim()

    obs0        = env._get_obs(x, goals, obstacles)
    obs_history = jnp.repeat(obs0[None], env.history_len, axis=0)
    act_history = jnp.zeros((env.history_len - 1, n_agents, env.action_dim))

    def body(carry, _):
        x, obs_history, act_history = carry

        obs_seq = obs_history.transpose(1, 0, 2)
        act_seq = act_history.transpose(1, 0, 2)
        u_nom   = env.u_nom_from_state(x, goals)

        out       = policy_apply_fn(policy_params, obs_seq, act_seq, u_nom, True)
        u_applied = jnp.clip(u_nom + action_scale * out.delta_u, action_lb, action_ub)

        x_next   = x + out.delta_x_hat
        obs_next = env._get_obs(x_next, goals, obstacles)

        new_obs_history = jnp.concatenate([obs_history[1:], obs_next[None]], axis=0)
        new_act_history = jnp.concatenate([act_history[1:], u_applied[None]], axis=0)

        is_unsafe = inside_obstacles(x_next[:, :2], obstacles, r=car_r)
        is_finish = jnp.linalg.norm(x_next[:, :2] - goals[:, :2], axis=-1) < car_r * 2

        cost   = is_unsafe.mean()
        reward = -(jnp.linalg.norm(u_applied - u_nom, axis=-1) ** 2).mean()

        return (x_next, new_obs_history, new_act_history), (is_unsafe, is_finish, reward, cost)

    _, (step_unsafe, step_finish, step_rewards, step_costs) = jax.lax.scan(
        body,
        (x, obs_history, act_history),
        xs=None,
        length=env.max_episode_steps,
    )
    return step_unsafe, step_finish, step_rewards, step_costs


def rollout_scaled(env, policy_apply_fn, policy_params, n_agents, horizon, key, action_scale=1.0):
    """Standard env.step() rollout with a CLI-controlled delta_u multiplier."""
    action_lb, action_ub = env.action_lim()
    key_x0, key = jax.random.split(key)
    init_graph  = env.reset(key_x0)

    def body(graph, _):
        u_first, _ = receding_horizon_act(
            policy_apply_fn=policy_apply_fn,
            policy_params=policy_params,
            graph=graph,
            env=env,
            n_agents=n_agents,
            horizon=horizon,
            deterministic=True,
            return_full_rollout=False,
        )
        x       = graph.type_states(type_idx=0, n_type=n_agents)
        goals   = graph.env_states.goal
        u_nom   = env.u_nom_from_state(x, goals)
        delta_u = u_first - u_nom
        action  = jnp.clip(u_nom + action_scale * delta_u, action_lb, action_ub)

        next_graph, reward, cost, done, _ = env.step(graph, action)
        return next_graph, (graph, action, reward, cost, done, next_graph)

    keys = jax.random.split(key, env.max_episode_steps)
    _, (graphs, actions, rewards, costs, dones, next_graphs) = \
        jax.lax.scan(body, init_graph, keys, length=env.max_episode_steps)
    return Rollout(graphs, actions, rewards, costs, dones, next_graphs)


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    cfg_path = os.path.join(path, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"config.yaml not found at {cfg_path}. "
            "Run with train_transformer.py (not --debug) to save the config."
        )
    with open(cfg_path, "r") as f:
        docs = list(yaml.load_all(f, Loader=yaml.UnsafeLoader))
    cfg = {}
    for doc in docs:
        if isinstance(doc, dict):
            cfg.update(doc)
    return cfg


def _pick_step(model_path: str, requested: int | None) -> int:
    if requested is not None:
        return requested
    entries = [e for e in os.listdir(model_path) if e.isdigit()]
    if not entries:
        raise RuntimeError(f"No numeric checkpoint directories found in {model_path}")
    return max(int(e) for e in entries)


# ── Main test function ────────────────────────────────────────────────────────

def test(args):
    if args.n_env is not None:
        args.epi = args.n_env

    assert args.env_id in ENV, (
        f"Unknown --env-id '{args.env_id}'. Valid: {list(ENV.keys())}"
    )

    stamp    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp_fn = datetime.datetime.now().strftime("%m%d-%H%M")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    pretrain_mode = args.pretrain_ckpt is not None

    if pretrain_mode:
        cfg         = {}
        history_len = 12
        horizon     = 6
        step        = "pretrain"
        print(f"> test_transformer_eval.py  [pretrain-eval mode]")
        print(f"  ckpt={args.pretrain_ckpt}")
    else:
        cfg         = _load_config(args.path)
        history_len = int(cfg.get("history_len", 12))
        horizon     = int(cfg.get("horizon",      6))

    EnvClass   = ENV[args.env_id]
    env_def    = _ENV_DEFAULTS.get(args.env_id, {})
    num_agents = args.num_agents if args.num_agents is not None else int(cfg.get("num_agents", 1))
    area_size  = args.area_size  if args.area_size  is not None else float(cfg.get("area_size", env_def.get("area_size", 4.0)))
    max_step   = args.max_step   if args.max_step   is not None else int(cfg.get("max_step",   256))
    n_obs      = args.n_obs  if args.n_obs  is not None else int(cfg.get("n_obs",  env_def.get("n_obs",  EnvClass.PARAMS.get("n_obs", 8))))
    n_rays     = args.n_rays if args.n_rays is not None else int(cfg.get("n_rays", env_def.get("n_rays", EnvClass.PARAMS.get("n_rays", 32))))

    # obs_len_range only applies to envs that support it (e.g. CrazyFlie)
    obs_len_range = None
    if args.env_id in _ENVS_WITH_OBS_LEN:
        if args.obs_len_range is not None:
            obs_len_range = list(args.obs_len_range)
        else:
            obs_len_range = list(cfg.get("obs_len_range", env_def.get("obs_len_range",
                                         EnvClass.PARAMS.get("obs_len_range"))))

    print(f"> test_transformer_eval.py")
    print(f"  env={args.env_id}  path={args.path}")
    print(f"  n_agents={num_agents}  area={area_size}  max_step={max_step}")
    print(f"  history_len={history_len}  horizon={horizon}")
    if obs_len_range is not None:
        print(f"  n_obs={n_obs}  n_rays={n_rays}  obs_len_range={obs_len_range}")
    else:
        print(f"  n_obs={n_obs}  n_rays={n_rays}")

    if args.use_dyn_head and args.env_id not in _ENVS_WITH_HISTORY:
        raise ValueError(
            f"--use-dyn-head requires history_len support; env '{args.env_id}' does not have it. "
            f"Valid envs: {sorted(_ENVS_WITH_HISTORY)}"
        )

    # ── Environment ───────────────────────────────────────────────────────────
    env = _make_env(
        env_id=args.env_id,
        num_agents=num_agents,
        area_size=area_size,
        max_step=max_step,
        history_len=history_len,
        n_obs=n_obs,
        n_rays=n_rays,
        use_alt_u_ref=args.use_alt_u_ref,
        obs_len_range=obs_len_range,
    )
    print(f"  obs_dim={env.obs_dim}  state_dim={env.state_dim}  action_dim={env.action_dim}")

    # ── Algorithm ────────────────────────────────────────────────────────────
    algo = BarrierFormer(
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=num_agents,
        gnn_layers=int(cfg.get("gnn_layers", 1)),
        batch_size=int(cfg.get("batch_size", 256)),
        buffer_size=int(cfg.get("buffer_size", 512)),
        horizon=horizon,
        lr_actor=float(cfg.get("lr_actor", 3e-5)),
        lr_cbf=float(cfg.get("lr_cbf",   3e-5)),
        alpha=float(cfg.get("alpha", 1.0)),
        eps=0.02,
        inner_epoch=int(cfg.get("inner_epoch", 8)),
        max_grad_norm=2.0,
        seed=int(cfg.get("seed", 0)),
        loss_action_coef=float(cfg.get("loss_action_coef", 0.001)),
        loss_unsafe_coef=float(cfg.get("loss_unsafe_coef", 1.0)),
        loss_safe_coef=float(cfg.get("loss_safe_coef",   1.0)),
        loss_dt_cbf_coef=float(cfg.get("loss_dt_cbf_coef", 0.2)),
        loss_dyn_coef=float(cfg.get("loss_dyn_coef", 1.0)),
        sqp_horizon_len=horizon,
        n_sqp_iter=int(cfg.get("n_sqp_iter", 3)),
        relax_penalty=float(cfg.get("relax_penalty", 1e3)),
        labeling_horizon=int(cfg.get("labeling_horizon", 32)),
        beta=float(cfg.get("beta", 10.0)),
        gamma=float(cfg.get("gamma", 0.0)),
    )

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if pretrain_mode:
        with open(args.pretrain_ckpt, "rb") as f:
            pretrained_params = pickle.load(f)
        algo.tf_actor_train_state = algo.tf_actor_train_state.replace(params=pretrained_params)
        print(f"  Pretrained weights loaded  (CBF-MLP is random — safety from imitation only)")
    else:
        model_path = os.path.join(args.path, "models")
        step = _pick_step(model_path, args.step)
        algo.load(model_path, step)

    # ── Build JIT-compiled rollout function ───────────────────────────────────
    if args.use_dyn_head:
        print(f"  Mode: DYNAMICS-HEAD world model  (x_{{t+1}} = x_t + delta_x_hat, no env.step)")
        def rollout_fn_single(params, key):
            return rollout_dyn_head(env, algo.tf_policy.apply, params, num_agents, key,
                                    action_scale=args.action_scale)
        rollout_fn = jax.jit(rollout_fn_single)
    else:
        def rollout_fn_single(params, key):
            return rollout_scaled(
                env,
                policy_apply_fn=algo.tf_policy.apply,
                policy_params=params,
                n_agents=num_agents,
                horizon=horizon,
                key=key,
                action_scale=args.action_scale,
            )
        rollout_fn = jax.jit(rollout_fn_single)

    if args.parallel:
        rollout_fn_batched = jax.jit(jax.vmap(rollout_fn_single, in_axes=(None, 0)))
        print(f"  [parallel] JIT+vmap compiled over {args.epi} episodes")

    finish_fn = jax.jit(jax_vmap(env.finish_mask))
    unsafe_fn = jax.jit(jax_vmap(env.collision_mask))

    # ── Key generation — support multiple seeds ───────────────────────────────
    active_seeds = args.seeds if args.seeds else [args.seed]
    n_seeds      = len(active_seeds)
    total_epi    = n_seeds * args.epi

    all_keys = jnp.concatenate([
        jr.split(jr.PRNGKey(s), 1_000)[: args.epi]
        for s in active_seeds
    ], axis=0)   # (total_epi, 2)

    print(f"  seeds={active_seeds}  epi_per_seed={args.epi}  total_epi={total_epi}")

    rewards, costs_sum, costs_max = [], [], []
    is_unsafes, is_finishes = [], []
    per_step_unsafes = []
    rollouts_list = []

    params = algo.tf_actor_train_state.params

    # ── Optional: pre-compute all rollouts in parallel with jax.vmap ─────────
    pre_rollouts = None

    if args.parallel:
        print(f"  [parallel] running {total_epi} episodes ({n_seeds} seeds × {args.epi}) with jax.vmap ...")
        t0 = time.time()
        batched    = rollout_fn_batched(params, all_keys)
        batched_np = jax.tree_util.tree_map(np.array, batched)
        print(f"  [parallel] done in {time.time() - t0:.1f}s")
        if args.use_dyn_head:
            pre_rollouts = [tuple(b[i] for b in batched_np) for i in range(total_epi)]
        else:
            pre_rollouts = [
                jax.tree_util.tree_map(lambda x, _i=i: x[_i], batched_np)
                for i in range(total_epi)
            ]

    for i_epi in range(total_epi):
        seed_id     = i_epi // args.epi
        epi_in_seed = i_epi % args.epi

        if args.use_dyn_head:
            if pre_rollouts is not None:
                step_unsafe, step_finish, step_rewards, step_costs = pre_rollouts[i_epi]
            else:
                step_unsafe, step_finish, step_rewards, step_costs = rollout_fn(params, all_keys[i_epi])
            step_unsafe  = np.array(step_unsafe)
            step_finish  = np.array(step_finish)
            epi_reward   = float(np.array(step_rewards).sum())
            epi_cost_sum = float(np.array(step_costs).sum())
            epi_cost_max = float(np.array(step_costs).max())
        else:
            if pre_rollouts is not None:
                ro = pre_rollouts[i_epi]
            else:
                ro = rollout_fn(params, all_keys[i_epi])
            rollouts_list.append(ro)
            epi_reward   = float(ro.rewards.sum())
            epi_cost_sum = float(ro.costs.sum())
            epi_cost_max = float(ro.costs.max())
            step_unsafe = np.array(unsafe_fn(ro.graph))
            step_finish = np.array(finish_fn(ro.graph))
            last_graph  = jax.tree_util.tree_map(lambda x: x[-1:], ro.next_graph)
            last_unsafe = np.array(unsafe_fn(last_graph))
            last_finish = np.array(finish_fn(last_graph))
            step_unsafe = np.concatenate([step_unsafe, last_unsafe], axis=0)
            step_finish = np.concatenate([step_finish, last_finish], axis=0)

        rewards.append(epi_reward)
        costs_sum.append(epi_cost_sum)
        costs_max.append(epi_cost_max)
        per_step_unsafes.append(step_unsafe)

        is_unsafe  = step_unsafe.max(axis=0)
        is_finish  = step_finish.max(axis=0)
        is_unsafes.append(is_unsafe)
        is_finishes.append(is_finish)

        safe_rate    = float(1 - is_unsafe.mean())
        finish_rate  = float(is_finish.mean())
        success_rate = float(((1 - is_unsafe) * is_finish).mean())

        print(f"  seed{active_seeds[seed_id]} epi {epi_in_seed:3d} | "
              f"reward {epi_reward:8.3f}  cost_sum {epi_cost_sum:7.3f}  cost_max {epi_cost_max:7.3f}  "
              f"safe {safe_rate*100:5.1f}%  finish {finish_rate*100:5.1f}%  "
              f"success {success_rate*100:5.1f}%")

    # ── Per-seed summary (within-seed std across episodes) ────────────────────
    is_unsafe_all = np.stack(is_unsafes)   # (total_epi, n_agents)
    is_finish_all = np.stack(is_finishes)  # (total_epi, n_agents)

    print(f"\n  ── Per-seed summary  "
          f"[env={args.env_id}  n_obs={n_obs}  n_rays={n_rays}  area={area_size}  ckpt={step}] ──")
    for i_seed, s in enumerate(active_seeds):
        sl = slice(i_seed * args.epi, (i_seed + 1) * args.epi)
        sm, ss, fm, fs, um, us = _stats_within_seed(is_unsafe_all[sl], is_finish_all[sl])
        print(f"  seed {s:2d}: "
              f"safe {sm*100:.2f}%±{ss*100:.2f}%  "
              f"finish {fm*100:.2f}%±{fs*100:.2f}%  "
              f"success {um*100:.2f}%±{us*100:.2f}%  "
              f"(std over {args.epi} episodes)")

    # ── Aggregate — std is ACROSS seeds (correct reportable variance) ─────────
    (safe_mean, safe_std,
     finish_mean, finish_std,
     success_mean, success_std,
     safe_ps, finish_ps, succ_ps) = _stats_cross_seed(
        is_unsafe_all, is_finish_all, args.epi, n_seeds)

    seeds_str = "-".join(str(s) for s in active_seeds)
    print(f"\n  ── Aggregate  "
          f"[{total_epi} epi  seeds={seeds_str}  env={args.env_id}  "
          f"n_obs={n_obs}  area={area_size}  ckpt={step}] ──")
    print(f"  NOTE: ± is std *across seeds* (correct reportable variance)")
    print(f"  reward  : {np.mean(rewards):.3f}  [{np.min(rewards):.3f}, {np.max(rewards):.3f}]")
    print(f"  cost_sum: {np.mean(costs_sum):.3f}  [{np.min(costs_sum):.3f}, {np.max(costs_sum):.3f}]")
    print(f"  cost_max: {np.mean(costs_max):.3f}  [{np.min(costs_max):.3f}, {np.max(costs_max):.3f}]")
    print(f"  safe    : {safe_mean*100:.2f}% ± {safe_std*100:.2f}%  ({n_seeds} seeds)")
    print(f"  finish  : {finish_mean*100:.2f}% ± {finish_std*100:.2f}%  ({n_seeds} seeds)")
    print(f"  success : {success_mean*100:.2f}% ± {success_std*100:.2f}%  ({n_seeds} seeds)")

    # ── Optional CSV log ──────────────────────────────────────────────────────
    if args.log:
        log_path = os.path.join(args.path, "test_log.csv")

        # One row per seed — std is within-seed (across episodes)
        for i_seed, s in enumerate(active_seeds):
            sl = slice(i_seed * args.epi, (i_seed + 1) * args.epi)
            sm, ss, fm, fs, um, us = _stats_within_seed(is_unsafe_all[sl], is_finish_all[sl])
            _append_log(
                log_path, stamp, args.env_id, step, num_agents, n_obs, n_rays,
                area_size, max_step,
                row_type="seed", seed_label=str(s),
                epi_count=args.epi,
                finish=fm, safe=sm, succ=um,
                finish_std=fs, safe_std=ss, succ_std=us,
            )

        # One aggregate row — std is across seeds (correct reportable metric)
        _append_log(
            log_path, stamp, args.env_id, step, num_agents, n_obs, n_rays,
            area_size, max_step,
            row_type="aggregate", seed_label=seeds_str,
            epi_count=total_epi,
            finish=finish_mean, safe=safe_mean, succ=success_mean,
            finish_std=finish_std, safe_std=safe_std, succ_std=success_std,
        )

        print(f"\n  Results appended to {log_path}")
        print(f"  CSV columns: timestamp | env_id | ckpt_step | n_agents | n_obs | n_rays | area_size | max_step")
        print(f"               row_type | seed | epi_count | finish% | safe% | success%")
        print(f"               finish_std% | safe_std% | success_std%")
        print(f"  seed rows    → std = within-seed (over episodes)")
        print(f"  aggregate row → std = across seeds (correct reportable variance)")

    # ── Optional video rendering ──────────────────────────────────────────────
    if args.no_video or args.use_dyn_head:
        if args.use_dyn_head and not args.no_video:
            print("  (video not available in --use-dyn-head mode — no real graph)")
        return

    videos_dir = pathlib.Path(args.path) / "videos"
    videos_dir.mkdir(exist_ok=True, parents=True)

    from barrierformer.env.base import RolloutResult
    from barrierformer.utils.utils import tree_index
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cbf_params = algo.cbf_mlp_train_state.params
    _CBF_RES   = 48

    @jax.jit
    def _h_at_agent(graph_t):
        state = graph_t.type_states(type_idx=0, n_type=num_agents)
        goal  = graph_t.env_states.goal
        obs   = env._get_obs(state, goal, graph_t.env_states.obstacle)
        return algo.cbf_mlp.apply(cbf_params, obs).squeeze()

    def _plot_h_vs_time(h_traj, unsafe_traj, save_path):
        T  = len(h_traj)
        ts = np.arange(T)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(ts, h_traj, color="#0068ff", linewidth=1.5, label="h(xₜ)")
        ax.axhline(0.0, color="k", linewidth=1.0, linestyle="--", label="h=0 boundary")
        unsafe_steps = np.where(np.array(unsafe_traj))[0]
        if len(unsafe_steps):
            ax.scatter(unsafe_steps, np.array(h_traj)[unsafe_steps],
                       color="red", zorder=5, s=30, label="unsafe (geometric)")
        ax.fill_between(ts, np.array(h_traj), 0,
                        where=np.array(h_traj) < 0,
                        alpha=0.25, color="red", label="h<0 region")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("h(xₜ)")
        ax.set_title("Learned CBF value along trajectory")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(save_path), dpi=120)
        plt.close(fig)

    epi_filter = None
    if args.epi_list:
        epi_filter = set(int(x) for x in args.epi_list.split(","))
        print(f"  Rendering only episodes: {sorted(epi_filter)}")

    for ii, (ro, is_unsafe) in enumerate(zip(rollouts_list, is_unsafes)):
        if epi_filter is not None and ii not in epi_filter:
            continue
        sr  = float((1 - is_unsafe).mean()) * 100
        fr  = float(is_finishes[ii].mean()) * 100
        suc = float(((1 - is_unsafe) * is_finishes[ii]).mean()) * 100
        vid_name = (f"{args.env_id}_n{num_agents}_step{step}_seed{args.seed}_epi{ii:02d}_"
                    f"sr{sr:.0f}_fr{fr:.0f}_suc{suc:.0f}")
        vid_path = videos_dir / f"{stamp_fn}_{vid_name}.mp4"

        T_steps  = ro.rewards.shape[0]
        T_reward = np.asarray(ro.rewards)
        T_cost   = np.asarray(ro.costs)
        if T_reward.ndim > 1:
            T_reward = T_reward.sum(axis=-1)
        if T_cost.ndim > 1:
            T_cost = T_cost.sum(axis=-1)

        h_traj = []
        for t in range(T_steps):
            g_t   = tree_index(ro.graph, t)
            h_val = float(np.array(_h_at_agent(g_t)).mean())
            h_traj.append(h_val)

        unsafe_traj = per_step_unsafes[ii][:T_steps].max(axis=-1)

        h_plot_path = videos_dir / f"{stamp_fn}_{vid_name}_h_traj.png"
        _plot_h_vs_time(h_traj, unsafe_traj, h_plot_path)
        print(f"  h-traj plot: {h_plot_path}")

        rr = RolloutResult(
            Tp1_graph=ro.graph,
            T_action=ro.actions,
            T_reward=T_reward,
            T_cost=T_cost,
            T_done=np.zeros(T_steps, dtype=bool),
            T_info={},
        )
        env.render_video(rr, vid_path, per_step_unsafes[ii], {}, dpi=args.dpi)
        print(f"  Video saved: {vid_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generic evaluator for transformer CBF checkpoint (all environments)"
    )
    parser.add_argument("--env-id",        type=str,   default="DubinsCar",
                        choices=list(ENV.keys()),
                        help="Environment to evaluate on (default: DubinsCar)")
    parser.add_argument("--path",          type=str,   default=None,
                        help="Run directory containing config.yaml and models/")
    parser.add_argument("--pretrain-ckpt", type=str,   default=None,
                        help="Path to a pretrained params_*.pkl to evaluate before CBF training")
    parser.add_argument("--use-dyn-head",  action="store_true", default=False,
                        help="Use dynamics head as world model: x_{t+1}=x_t+delta_x_hat (no env.step). "
                             f"Only valid for: {sorted(_ENVS_WITH_HISTORY)}")
    parser.add_argument("--step",          type=int,   default=None,
                        help="Checkpoint step (default: latest)")
    parser.add_argument("-n", "--num-agents", type=int, default=1)
    parser.add_argument("--area-size",     type=float, default=None)
    parser.add_argument("--max-step",      type=int,   default=None)
    parser.add_argument("--n-env",         type=int,   default=None,
                        help="Number of parallel environments / episodes per seed (preferred over --epi)")
    parser.add_argument("--epi",           type=int,   default=32,
                        help="Number of test episodes per seed (default: 32). --n-env overrides this.")
    parser.add_argument("--seed",          type=int,   default=1234,
                        help="Single random seed (default). Use --seeds for multi-seed eval.")
    parser.add_argument("--seeds",         type=int,   nargs="+", default=None,
                        help="Multiple seeds, e.g. --seeds 0 1 2. "
                             "Overrides --seed. Reports per-seed + aggregate stats.")
    parser.add_argument("--cpu",           action="store_true", default=False)
    parser.add_argument("--debug",         action="store_true", default=False,
                        help="Disable JIT for debugging")
    parser.add_argument("--parallel",      action="store_true", default=False,
                        help="Run all episodes in parallel with jax.vmap")
    parser.add_argument("--no-video",      action="store_true", default=False)
    parser.add_argument("--cbf-video",     action="store_true", default=False,
                        help="Overlay CBF h-value contour on video (slower)")
    parser.add_argument("--epi-list",      type=str,   default=None,
                        help="Comma-separated episode indices to render (e.g., '0,3'). Default: all.")
    parser.add_argument("--log",           action="store_true", default=False,
                        help="Append results to test_log.csv in the run directory. "
                             "Writes a header on first use. Columns include env_id, n_obs, n_rays, "
                             "area_size. One row per seed (within-seed std over episodes) plus one "
                             "aggregate row (std across seeds — the correct reportable variance).")
    parser.add_argument("--action-scale",  type=float, default=1.0,
                        help="Multiplier on delta_u: u_applied = u_nom + scale * delta_u (default: 1.0)")
    parser.add_argument("--n-obs",         type=int,   default=None,
                        help="Number of obstacles (overrides config.yaml)")
    parser.add_argument("--n-rays",        type=int,   default=None,
                        help="Number of LiDAR rays (overrides config.yaml)")
    parser.add_argument("--use-alt-u-ref", action="store_true", default=False,
                        help="Use alternative u_ref nominal controller (DubinsCar only)")
    parser.add_argument("--obs-len-range", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Obstacle diameter range in metres (CrazyFlie only), e.g. "
                             "--obs-len-range 0.2 0.7. Default for CrazyFlie: [0.2, 0.7]. "
                             "Sphere radius is sampled uniformly from [MIN/2, MAX/2].")
    parser.add_argument("--dpi",           type=int,   default=100)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()

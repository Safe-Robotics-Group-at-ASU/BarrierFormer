"""
test_transformer_progressive_remove_3step_all.py
================================================
Three-step progressive-removal evaluator — compatible with all environments:
DoubleIntegrator, DubinsCar, CrazyFlie.

Same idea as test_transformer_progressive_remove_3step.py: each (l) is
evaluated at THREE densities — ρ=1.0 → 0.75 → 0.50 — using one shared
environment instance per (seed, episode).  The dense (ρ=1.0) graph is reset
once; medium and sparse graphs are formed by sentinelizing the trailing
obstacles, so all three rollouts share the same start, goal, kept-obstacle
layout, RNG key, and policy.

Obstacle sentinelization is env-aware:
  • DubinsCar / DoubleIntegrator → Rectangle obstacles (2-D center + w/h/θ)
  • CrazyFlie                    → Sphere obstacles   (3-D center + radius)

Default triples per env
-----------------------
DubinsCar / DoubleIntegrator  (2-D density ρ = n / l²):
    l=8 :  64 → 48 → 32
    l=6 :  36 → 27 → 18
    l=4 :  16 → 12 →  8

CrazyFlie  (area_size = 3.0 m, default n_obs = 6):
    l=3 :   6 →  4 →  2

Usage:
    python test_transformer_progressive_remove_3step_all.py \\
        --env-id DoubleIntegrator \\
        --path ./Trained-Weights/DI/DI-Sim/ \\
        --seeds 0 1 2 --n-env 32 --parallel --log

    python test_transformer_progressive_remove_3step_all.py \\
        --env-id CrazyFlie \\
        --path ./Trained-Weights/CF/CF-Sim/ \\
        --seeds 0 1 2 --n-env 32 --parallel --log
"""

import argparse
import datetime
import os
import time
from typing import List, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from barrierformer.env import ENV
from barrierformer.env.obstacle import Rectangle, Sphere
from barrierformer.algo.barrierformer import BarrierFormer

from test_transformer_eval import (
    _make_env, _load_config, _pick_step,
    _stats_cross_seed, _append_log,
    _ENV_DEFAULTS, _ENVS_WITH_OBS_LEN,
    _graph_with_obstacles, rollout_from_graph,
)


# ── Default triples per env ────────────────────────────────────────────────────

# (l, n_dense, n_mid, n_sparse) — ρ=1.0 → 0.75 → 0.50 for 2-D envs
_DEFAULT_TRIPLES_2D: List[Tuple[float, int, int, int]] = [
  #  (8.0, 64, 48, 32),
    (6.0, 36, 27, 18),
    (4.0, 16, 12,  8),
]

# For CrazyFlie: area_size=3.0, volumetric density rho = n_obs / 3^3 = n_obs/27
# Training density rho=1.0 (n_obs=27) → 0.75 (n_obs=20) → 0.50 (n_obs=13)
_DEFAULT_TRIPLES_CF: List[Tuple[float, int, int, int]] = [
    (3.0, 27, 20, 13),
]

_DEFAULT_TRIPLES_BY_ENV = {
    "CrazyFlie": _DEFAULT_TRIPLES_CF,
}


def _default_triples(env_id: str) -> List[Tuple[float, int, int, int]]:
    return _DEFAULT_TRIPLES_BY_ENV.get(env_id, _DEFAULT_TRIPLES_2D)


# ── Episode length defaults ────────────────────────────────────────────────────

DEFAULT_MAX_STEP = 1024

def _max_steps_for(override: int | None) -> int:
    return override if override is not None else DEFAULT_MAX_STEP


# ── Obstacle sentinelization ───────────────────────────────────────────────────

def _sentinelize_rectangle(obstacles, n_dense: int, n_keep: int, area_size: float):
    """Sentinelize trailing Rectangle obstacles (DubinsCar / DoubleIntegrator)."""
    far = area_size * 100.0
    idx = jnp.arange(n_dense)
    keep = idx < n_keep
    far_center = jnp.array([far, far], dtype=obstacles.center.dtype)
    new_centers = jnp.where(keep[:, None], obstacles.center, far_center[None, :])
    rebuild = jax.vmap(Rectangle.create)
    return rebuild(new_centers, obstacles.width, obstacles.height, obstacles.theta)


def _sentinelize_sphere(obstacles, n_dense: int, n_keep: int, area_size: float):
    """Sentinelize trailing Sphere obstacles (CrazyFlie)."""
    far = area_size * 100.0
    idx = jnp.arange(n_dense)
    keep = idx < n_keep
    far_center = jnp.array([far, far, far], dtype=obstacles.center.dtype)
    new_centers = jnp.where(keep[:, None], obstacles.center, far_center[None, :])
    rebuild = jax.vmap(Sphere.create)
    return rebuild(new_centers, obstacles.radius)


def _sentinelize(env_id: str, obstacles, n_dense: int, n_keep: int, area_size: float):
    if env_id == "CrazyFlie":
        return _sentinelize_sphere(obstacles, n_dense, n_keep, area_size)
    return _sentinelize_rectangle(obstacles, n_dense, n_keep, area_size)


# ── Parse custom triples ───────────────────────────────────────────────────────

def _parse_triples(raw: str) -> List[Tuple[float, int, int, int]]:
    """Parse '--triples "8,64,48,32 6,36,27,18 4,16,12,8"'."""
    out = []
    for tok in raw.split():
        parts = tok.split(",")
        if len(parts) != 4:
            raise ValueError(f"each triple needs L,DENSE,MID,SPARSE; got '{tok}'")
        l_v, d, m, s = float(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        if not (d >= m >= s >= 0):
            raise ValueError(f"need DENSE >= MID >= SPARSE >= 0 in '{tok}'")
        out.append((l_v, d, m, s))
    return out


# ── Per-triple evaluator ───────────────────────────────────────────────────────

def _evaluate_triple(args, cfg, algo, l_val, n_dense, n_mid, n_sparse,
                     history_len, horizon, obs_len_range):
    n_rays   = args.n_rays if args.n_rays is not None else int(cfg.get("n_rays", 32))
    n_agents = args.num_agents if args.num_agents is not None else int(cfg.get("num_agents", 1))
    max_step = _max_steps_for(args.max_step)

    rho = lambda n: n / (l_val ** 2)
    print(f"\n=== l={l_val:.0f}   "
          f"dense {n_dense} (ρ={rho(n_dense):.2f}) → "
          f"mid {n_mid} (ρ={rho(n_mid):.2f}) → "
          f"sparse {n_sparse} (ρ={rho(n_sparse):.2f})   max_step={max_step} ===")

    env = _make_env(
        env_id=args.env_id, num_agents=n_agents, area_size=float(l_val),
        max_step=max_step, history_len=history_len,
        n_obs=n_dense, n_rays=n_rays,
        use_alt_u_ref=args.use_alt_u_ref,
        obs_len_range=obs_len_range,
    )

    reset_fn = jax.jit(env.reset)

    def make_dense_graph(key):
        return reset_fn(key)

    def make_graph_at(dense_graph, n_keep):
        new_obs = _sentinelize(
            args.env_id,
            dense_graph.env_states.obstacle, n_dense, n_keep, float(l_val))
        return _graph_with_obstacles(env, dense_graph, new_obs)

    make_mid_graph    = jax.jit(lambda dg: make_graph_at(dg, n_mid))
    make_sparse_graph = jax.jit(lambda dg: make_graph_at(dg, n_sparse))

    def rollout_fn_single(params, init_graph, key):
        return rollout_from_graph(
            env,
            policy_apply_fn=algo.tf_policy.apply,
            policy_params=params,
            n_agents=n_agents,
            horizon=horizon,
            init_graph=init_graph,
            key=key,
            action_scale=args.action_scale,
        )
    rollout_fn = jax.jit(rollout_fn_single)

    finish_fn = jax.jit(jax.vmap(env.finish_mask))
    unsafe_fn = jax.jit(jax.vmap(env.collision_mask))

    active_seeds = args.seeds if args.seeds else [args.seed]
    n_seeds      = len(active_seeds)
    epi_per_seed = args.n_env if args.n_env is not None else args.epi
    total_epi    = n_seeds * epi_per_seed

    all_keys = jnp.concatenate([
        jr.split(jr.PRNGKey(s), 1_000)[:epi_per_seed]
        for s in active_seeds
    ], axis=0)

    if args.parallel:
        rollout_fn_batched  = jax.jit(jax.vmap(rollout_fn_single, in_axes=(None, 0, 0)))
        make_dense_batched  = jax.jit(jax.vmap(make_dense_graph))
        make_mid_batched    = jax.jit(jax.vmap(make_mid_graph))
        make_sparse_batched = jax.jit(jax.vmap(make_sparse_graph))

    params = algo.tf_actor_train_state.params

    print(f"   running {total_epi} episodes ({n_seeds} seeds × {epi_per_seed}) "
          f"at three densities (same start/goal/seed across all three)")

    t0 = time.time()
    if args.parallel:
        dense_graphs  = make_dense_batched(all_keys)
        mid_graphs    = make_mid_batched(dense_graphs)
        sparse_graphs = make_sparse_batched(dense_graphs)
    print(f"   graph construction: {time.time() - t0:.1f}s")

    def _split_batched(batched):
        batched_np = jax.tree_util.tree_map(np.array, batched)
        return [jax.tree_util.tree_map(lambda x, _i=i: x[_i], batched_np)
                for i in range(total_epi)]

    def _run(label, graphs_batched, build_one):
        t0 = time.time()
        if args.parallel:
            ros = _split_batched(rollout_fn_batched(params, graphs_batched, all_keys))
        else:
            ros = [rollout_fn(params, build_one(make_dense_graph(all_keys[i])), all_keys[i])
                   for i in range(total_epi)]
        print(f"   [{label:>6s}] rollouts done in {time.time() - t0:.1f}s")
        return ros

    dense_ros  = _run("dense",  dense_graphs  if args.parallel else None, lambda dg: dg)
    mid_ros    = _run("mid",    mid_graphs    if args.parallel else None,
                      lambda dg: make_graph_at(dg, n_mid))
    sparse_ros = _run("sparse", sparse_graphs if args.parallel else None,
                      lambda dg: make_graph_at(dg, n_sparse))

    def _flags_from_rollouts(ros, metric_steps: int):
        is_unsafe_list, is_finish_list = [], []
        for ro in ros:
            step_unsafe = np.array(unsafe_fn(ro.graph))
            step_finish = np.array(finish_fn(ro.graph))
            last_graph  = jax.tree_util.tree_map(lambda x: x[-1:], ro.next_graph)
            last_unsafe = np.array(unsafe_fn(last_graph))
            last_finish = np.array(finish_fn(last_graph))
            step_unsafe = np.concatenate([step_unsafe, last_unsafe], axis=0)
            step_finish = np.concatenate([step_finish, last_finish], axis=0)
            cutoff = min(metric_steps + 1, step_unsafe.shape[0])
            step_unsafe = step_unsafe[:cutoff]
            step_finish = step_finish[:cutoff]
            is_unsafe_list.append(step_unsafe.max(axis=0))
            is_finish_list.append(step_finish.max(axis=0))
        return np.stack(is_unsafe_list), np.stack(is_finish_list)

    dense_steps  = _max_steps_for(args.max_step)
    mid_steps    = _max_steps_for(args.max_step)
    sparse_steps = _max_steps_for(args.max_step)
    if len({dense_steps, mid_steps, sparse_steps}) > 1:
        print(f"   metric windows: dense={dense_steps}, mid={mid_steps}, sparse={sparse_steps}")

    dense_unsafe,  dense_finish  = _flags_from_rollouts(dense_ros,  dense_steps)
    mid_unsafe,    mid_finish    = _flags_from_rollouts(mid_ros,    mid_steps)
    sparse_unsafe, sparse_finish = _flags_from_rollouts(sparse_ros, sparse_steps)

    def _summary(label, n_obs, unsafe_arr, finish_arr):
        (safe_m, safe_s, fin_m, fin_s, succ_m, succ_s,
         _, _, _) = _stats_cross_seed(unsafe_arr, finish_arr, epi_per_seed, n_seeds)
        print(f"   {label:6s} (n_obs={n_obs:3d}, ρ={rho(n_obs):.2f}): "
              f"safe {safe_m*100:6.2f}%±{safe_s*100:5.2f}%   "
              f"finish {fin_m*100:6.2f}%±{fin_s*100:5.2f}%   "
              f"success {succ_m*100:6.2f}%±{succ_s*100:5.2f}%")
        return safe_m, safe_s, fin_m, fin_s, succ_m, succ_s

    dense_stats  = _summary("dense",  n_dense,  dense_unsafe,  dense_finish)
    mid_stats    = _summary("mid",    n_mid,    mid_unsafe,    mid_finish)
    sparse_stats = _summary("sparse", n_sparse, sparse_unsafe, sparse_finish)

    def _flips(other_unsafe):
        return int(np.sum(dense_unsafe.max(axis=-1) != other_unsafe.max(axis=-1)))
    print(f"   safety-outcome flips vs. dense:  mid {_flips(mid_unsafe):>3d}/{total_epi}, "
          f"sparse {_flips(sparse_unsafe):>3d}/{total_epi}")

    return {
        "l": float(l_val),
        "counts": (n_dense, n_mid, n_sparse),
        "max_step": int(max_step),
        "metric_steps": {"dense": dense_steps, "mid": mid_steps, "sparse": sparse_steps},
        "n_agents": int(n_agents),
        "n_rays": int(n_rays),
        "epi_per_seed": int(epi_per_seed),
        "n_seeds": int(n_seeds),
        "seeds": active_seeds,
        "stats": {"dense": dense_stats, "mid": mid_stats, "sparse": sparse_stats},
        "flips": {"mid": _flips(mid_unsafe), "sparse": _flips(sparse_unsafe)},
        "total_epi": int(total_epi),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Three-step progressive-removal evaluator — all environments "
                    "(DoubleIntegrator, DubinsCar, CrazyFlie)."
    )
    parser.add_argument("--env-id", type=str, default="DoubleIntegrator",
                        choices=list(ENV.keys()))
    parser.add_argument("--path",   type=str, required=True)
    parser.add_argument("--step",   type=int, default=None)
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None,
                        help="Override episode length for all configs.")
    parser.add_argument("--n-env",  type=int, default=32)
    parser.add_argument("--epi",    type=int, default=32)
    parser.add_argument("--seed",   type=int, default=1234)
    parser.add_argument("--seeds",  type=int, nargs="+", default=None,
                        help="Multi-seed eval, e.g. --seeds 0 1 2.")
    parser.add_argument("--n-rays", type=int, default=None)
    parser.add_argument("--cpu",    action="store_true", default=False)
    parser.add_argument("--debug",  action="store_true", default=False)
    parser.add_argument("--parallel", action="store_true", default=False)
    parser.add_argument("--log",    action="store_true", default=False)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--use-alt-u-ref", action="store_true", default=False,
                        help="Use alternative u_ref nominal controller (DubinsCar only).")
    parser.add_argument("--obs-len-range", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Obstacle diameter range in metres (CrazyFlie only), e.g. "
                             "--obs-len-range 0.2 0.7. Default: [0.2, 0.7].")
    parser.add_argument("--triples", type=str, default=None,
                        help='Custom triples "L,DENSE,MID,SPARSE" space-separated, '
                             'e.g. "8,64,48,32 6,36,27,18 4,16,12,8".')

    args = parser.parse_args()

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    triples = _parse_triples(args.triples) if args.triples else _default_triples(args.env_id)

    cfg         = _load_config(args.path)
    history_len = int(cfg.get("history_len", 12))
    horizon     = int(cfg.get("horizon",      6))

    # Resolve obs_len_range (CrazyFlie only)
    env_def = _ENV_DEFAULTS.get(args.env_id, {})
    obs_len_range = None
    if args.env_id in _ENVS_WITH_OBS_LEN:
        if args.obs_len_range is not None:
            obs_len_range = list(args.obs_len_range)
        else:
            obs_len_range = list(cfg.get("obs_len_range", env_def.get("obs_len_range",
                                         ENV[args.env_id].PARAMS.get("obs_len_range"))))

    # Build proto env for BarrierFormer shape inference
    n_agents_proto = args.num_agents if args.num_agents is not None else int(cfg.get("num_agents", 1))
    n_rays_proto   = args.n_rays if args.n_rays is not None else int(cfg.get("n_rays", 32))
    proto_env = _make_env(
        env_id=args.env_id,
        num_agents=n_agents_proto,
        area_size=float(triples[0][0]),
        max_step=_max_steps_for(args.max_step),
        history_len=history_len,
        n_obs=int(triples[0][1]),
        n_rays=n_rays_proto,
        use_alt_u_ref=args.use_alt_u_ref,
        obs_len_range=obs_len_range,
    )

    algo = BarrierFormer(
        env=proto_env,
        node_dim=proto_env.node_dim, edge_dim=proto_env.edge_dim,
        state_dim=proto_env.state_dim, action_dim=proto_env.action_dim,
        n_agents=proto_env.num_agents,
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
    model_path = os.path.join(args.path, "models")
    step = _pick_step(model_path, args.step)
    algo.load(model_path, step)
    print(f"> 3-step progressive-removal eval  env={args.env_id}  ckpt_step={step}")
    print(f"  path={args.path}")
    print(f"  history_len={history_len}  horizon={horizon}")
    if obs_len_range is not None:
        print(f"  obs_len_range={obs_len_range}")
    print(f"  triples: {triples}")

    stamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = []
    for l_val, n_dense, n_mid, n_sparse in triples:
        results.append(_evaluate_triple(
            args, cfg, algo, l_val, n_dense, n_mid, n_sparse,
            history_len, horizon, obs_len_range))

    # ── Final summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 102)
    print(f"  3-step progressive-removal summary   env={args.env_id}   "
          f"ckpt={step}   {stamp}")
    print("=" * 102)
    print(f"  {'l':>3}  {'counts (D→M→S)':<16}  "
          f"{'mode':>6}  {'ρ':>5}  {'T':>5}  "
          f"{'Safety%':>14}  {'Reaching%':>14}  {'Success%':>14}  flips_vs_dense")
    print("-" * 102)
    for r in results:
        nd, nm, ns = r["counts"]
        for label, n in (("dense", nd), ("mid", nm), ("sparse", ns)):
            stats = r["stats"][label]
            safe_m, safe_s, fin_m, fin_s, succ_m, succ_s = stats
            flips_str = ("" if label == "dense"
                         else f"{r['flips'][label]:>3d}/{r['total_epi']}")
            print(f"  {r['l']:>3.0f}  {nd:>3d}→{nm:>3d}→{ns:<3d}{'':>5}  "
                  f"{label:>6s}  {n / (r['l']**2):>5.2f}  "
                  f"{r['metric_steps'][label]:>5d}  "
                  f"{safe_m*100:>7.2f} ± {safe_s*100:<3.2f}  "
                  f"{fin_m *100:>7.2f} ± {fin_s *100:<3.2f}  "
                  f"{succ_m*100:>7.2f} ± {succ_s*100:<3.2f}  "
                  f"{flips_str:>14s}")
    print("=" * 102)

    if args.log:
        log_path = os.path.join(args.path, "test_log.csv")
        for r in results:
            nd, nm, ns = r["counts"]
            for label, n in (("dense", nd), ("mid", nm), ("sparse", ns)):
                safe_m, safe_s, fin_m, fin_s, succ_m, succ_s = r["stats"][label]
                seeds_str = "-".join(str(s) for s in r["seeds"])
                _append_log(
                    log_path, stamp, args.env_id, step,
                    r["n_agents"], n, r["n_rays"], r["l"],
                    r["metric_steps"][label],
                    row_type=f"progrem3_{label}", seed_label=seeds_str,
                    epi_count=r["total_epi"],
                    finish=fin_m, safe=safe_m, succ=succ_m,
                    finish_std=fin_s, safe_std=safe_s, succ_std=succ_s,
                )
        print(f"  Results appended to {log_path} "
              f"(row_type = progrem3_dense / progrem3_mid / progrem3_sparse)")


if __name__ == "__main__":
    main()

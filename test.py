"""Evaluate a trained BarrierFormer checkpoint (DI, DC, CF): normal, generalization and robustness tests."""

import argparse
import os
import sys

ENVS = {"DI": "DoubleIntegrator", "DC": "DubinsCar", "CF": "CrazyFlie"}


def parse_args():
    """Command-line arguments."""
    ap = argparse.ArgumentParser(description="Evaluate a BarrierFormer checkpoint (normal / generalization / robustness).")
    ap.add_argument("--env", required=True, choices=list(ENVS) + list(ENVS.values()),
                    help="DI | DC | CF (or DoubleIntegrator | DubinsCar | CrazyFlie)")
    ap.add_argument("--ckpt", required=True, help="checkpoint folder with config.yaml and models/<step>/")
    ap.add_argument("--mode", choices=["normal", "generalization", "robustness"], default="normal")
    ap.add_argument("--step", type=int, default=1000, help="checkpoint step (default 1000)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="evaluation seeds (default 0 1 2)")
    ap.add_argument("--epi", type=int, default=32, help="episodes per seed (default 32)")
    ap.add_argument("--max-step", type=int, default=None,
                    help="episode length (default: 256 for normal/robustness, 1024 for generalization)")
    ap.add_argument("--triples", type=str, default=None,
                    help='generalization cells "l,M_dense,M_mid,M_sparse ..." (default: the paper setting per env)')
    ap.add_argument("--obs-len-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="obstacle size range")
    ap.add_argument("--noise-scale", choices=["state", "absolute"], default="state",
                    help="robustness noise: state = sigma*[4,4,0.5,0.5] (paper); absolute = sigma for all dims")
    ap.add_argument("--show-failures", action="store_true", help="list failed episodes (seed, episode, C/T)")
    ap.add_argument("--csv", type=str, default=None, help="also write the results table to this CSV file")
    ap.add_argument("--gpu", action="store_true", help="run on GPU instead of CPU")
    return ap.parse_args()


ARGS = parse_args() if __name__ == "__main__" else None
if ARGS is not None and not ARGS.gpu:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import pickle

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import yaml

from barrierformer.algo.gcbf_plus import GCBFPlus
from barrierformer.algo.receding_horizon import receding_horizon_act
from barrierformer.env.obstacle import Rectangle, Sphere
from barrierformer.utils.utils import jax_vmap

DEFAULT_TRIPLES = {
    "DoubleIntegrator": [(6.0, 36, 27, 18), (4.0, 16, 12, 8)],
    "DubinsCar": [(6.0, 36, 27, 18), (4.0, 16, 12, 8)],
    "CrazyFlie": [(3.0, 27, 20, 13), (4.0, 64, 48, 32)],
}
ALPHA_VEL = [0.8, 0.9, 1.0, 1.1, 1.2]
SIGMAS = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01]


def load_config(path):
    """Load the run's config.yaml."""
    with open(os.path.join(path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
    return cfg if isinstance(cfg, dict) else vars(cfg)


def infer_architecture(params):
    """Transformer geometry read from the checkpoint weights."""
    flat = {jtu.keystr(k): np.shape(v) for k, v in jtu.tree_flatten_with_path(params)[0]}

    def shape(key):
        for k, v in flat.items():
            if k.endswith(key):
                return v
        raise KeyError(key)

    hd = shape("['transformer']['ln_f']['scale']")[0]
    return dict(
        hidden_dim=int(hd),
        num_layers=len({k.split("blocks_")[1].split("'")[0] for k in flat if "blocks_" in k}),
        mlp_ratio=float(shape("['transformer']['blocks_0']['ffn']['fc1']['kernel']")[1] / hd),
        action_head_hidden_dim=int(shape("['action_head']['hidden_layers_0']['kernel']")[1]),
        action_head_num_layers=len({k.split("hidden_layers_")[1].split("'")[0]
                                    for k in flat if "['action_head']['hidden_layers_" in k}),
        dynamics_head_hidden_dim=int(shape("['dynamics_head']['first_layer']['kernel']")[1]),
        dynamics_head_num_layers=len({k.split("hidden_layers_")[1].split("'")[0]
                                      for k in flat if "['dynamics_head']['hidden_layers_" in k}),
        history_len=int((shape("['transformer']['pos_embed']['embedding']")[0] - 1) // 2 + 1),
        obs_dim=int(shape("['transformer']['obs_embed']['kernel']")[0]),
    )


def env_class(env_id):
    """Environment class for an env id."""
    if env_id == "DoubleIntegrator":
        from barrierformer.env.double_integrator import DoubleIntegrator as E
    elif env_id == "DubinsCar":
        from barrierformer.env.dubins_car import DubinsCar as E
    else:
        from barrierformer.env.crazyflie import CrazyFlie as E
    return E


def env_params(env_id, cfg, n_obs, obs_len_range=None):
    """Environment parameters from the training config."""
    E = env_class(env_id)
    if env_id == "DoubleIntegrator":
        extra = {"lidar_ray_order": bool(cfg.get("lidar_ray_order", False))}
    elif env_id == "DubinsCar":
        use_orig = cfg.get("use_orig_u_ref", None)
        extra = {"lidar_ray_order": bool(cfg.get("lidar_ray_order", False)),
                 "use_orig_u_ref": True if use_orig is None else bool(use_orig),
                 "state_obs": cfg.get("state_obs", "raw"),
                 "pid_k_v": float(cfg.get("pid_k_v") or 2.3)}
    else:
        extra = {"lidar_encoding": cfg.get("lidar_encoding") or "sector"}
    params = {**E.PARAMS, "n_rays": int(cfg.get("n_rays", 32)), "n_obs": int(n_obs), **extra}
    if obs_len_range is not None:
        params["obs_len_range"] = list(obs_len_range)
    return params


def make_env(env_id, cfg, history_len, area, n_obs, max_step, obs_len_range=None, cls=None, **kw):
    """Create the test environment."""
    E = cls or env_class(env_id)
    return E(num_agents=int(cfg.get("num_agents", 1)), area_size=float(area), max_step=int(max_step),
             history_len=history_len, params=env_params(env_id, cfg, n_obs, obs_len_range), **kw)


def build_algo(env, cfg, arch, horizon):
    """Build the algorithm and load the checkpoint weights."""
    return GCBFPlus(
        env=env, node_dim=env.node_dim, edge_dim=env.edge_dim, state_dim=env.state_dim,
        action_dim=env.action_dim, n_agents=env.num_agents, gnn_layers=int(cfg.get("gnn_layers", 1)),
        batch_size=int(cfg.get("batch_size", 256)), buffer_size=int(cfg.get("buffer_size", 512)),
        horizon=horizon, lr_actor=float(cfg.get("lr_actor", 3e-5)), lr_cbf=float(cfg.get("lr_cbf", 3e-5)),
        alpha=float(cfg.get("alpha", 1.0)), eps=0.02, inner_epoch=int(cfg.get("inner_epoch", 8)),
        max_grad_norm=2.0, seed=int(cfg.get("seed", 0)),
        loss_action_coef=float(cfg.get("loss_action_coef", 0.001)),
        loss_unsafe_coef=float(cfg.get("loss_unsafe_coef", 1.0)),
        loss_safe_coef=float(cfg.get("loss_safe_coef", 1.0)),
        loss_dt_cbf_coef=float(cfg.get("loss_dt_cbf_coef", 0.2)),
        loss_dyn_coef=float(cfg.get("loss_dyn_coef", 1.0)),
        sqp_horizon_len=horizon, n_sqp_iter=int(cfg.get("n_sqp_iter", 3)),
        relax_penalty=float(cfg.get("relax_penalty", 1e3)),
        labeling_horizon=int(cfg.get("labeling_horizon", 32)),
        beta=float(cfg.get("beta", 10.0)), gamma=float(cfg.get("gamma", 0.0)),
        tf_hidden_dim=arch["hidden_dim"], tf_num_heads=int(cfg.get("num_heads", 2)),
        tf_num_layers=arch["num_layers"], tf_mlp_ratio=arch["mlp_ratio"], tf_dropout=0.0,
        action_head_hidden_dim=arch["action_head_hidden_dim"],
        action_head_num_layers=arch["action_head_num_layers"],
        dynamics_head_hidden_dim=arch["dynamics_head_hidden_dim"],
        dynamics_head_num_layers=arch["dynamics_head_num_layers"])


def episode_keys(seeds, epi):
    """Random keys for all episodes of all seeds."""
    return jnp.concatenate([jr.split(jr.PRNGKey(s), 1000)[:epi] for s in seeds], axis=0)


def rollout_from_graph(env, policy_apply, params, horizon, init_graph, key):
    """Closed-loop episode: plan H steps, apply only the first action to env.step."""
    _key_x0, key = jax.random.split(key)

    def body(graph, key_):
        action, _ = receding_horizon_act(
            policy_apply_fn=policy_apply, policy_params=params, graph=graph, env=env,
            n_agents=env.num_agents, horizon=horizon, deterministic=True, return_full_rollout=False)
        env._noise_key = jr.fold_in(key_, 0x5EED)
        next_graph, reward, cost, _, _ = env.step(graph, action)
        env._noise_key = None
        return next_graph, (graph, next_graph, reward, cost)

    keys = jax.random.split(key, env.max_episode_steps)
    _, (graphs, next_graphs, rewards, costs) = jax.lax.scan(body, init_graph, keys, length=env.max_episode_steps)
    return graphs, next_graphs, rewards.sum(), costs.sum()


def run(env, algo, horizon, init_graphs, keys):
    """Run all episodes in parallel; returns per-episode (unsafe, finish, reward, cost) arrays of shape (n_epi,)."""
    params = algo.tf_actor_train_state.params
    roll = jax.jit(jax.vmap(lambda g, k: rollout_from_graph(env, algo.tf_policy.apply, params, horizon, g, k)))
    graphs, next_graphs, reward, cost = jtu.tree_map(np.asarray, roll(init_graphs, keys))
    unsafe_fn, finish_fn = jax.jit(jax_vmap(env.collision_mask)), jax.jit(jax_vmap(env.finish_mask))
    unsafe, finish = [], []
    for i in range(len(keys)):
        g = jtu.tree_map(lambda x: x[i], graphs)
        last = jtu.tree_map(lambda x: x[i][-1:], next_graphs)
        unsafe.append(np.concatenate([np.asarray(unsafe_fn(g)), np.asarray(unsafe_fn(last))], 0).max(0))
        finish.append(np.concatenate([np.asarray(finish_fn(g)), np.asarray(finish_fn(last))], 0).max(0))
    return np.stack(unsafe).astype(float).max(-1), np.stack(finish).astype(float).max(-1), reward, cost


def print_episodes(unsafe, finish, reward, cost, seeds, epi):
    """Print reward, cost, safe, reach and success of every episode."""
    for i in range(len(unsafe)):
        safe, reach = int(1 - unsafe[i]), int(finish[i])
        print(f"seed {seeds[i // epi]} | episode {i % epi:3d} | reward: {reward[i]:9.3f} | cost: {cost[i]:7.3f} | "
              f"safe: {safe} | reach: {reach} | success: {safe * reach}", flush=True)


def reset_graphs(env, keys):
    """Reset one environment per key."""
    return jax.jit(jax.vmap(lambda k: env.reset(jax.random.split(k)[0])))(keys)


def remove_obstacles(env_id, env, graphs, n_dense, n_keep, area):
    """Move extra obstacles far away to lower the density and re-seed the history."""
    if n_keep == n_dense:
        return graphs
    far = area * 100.0

    def one(graph):
        obs = graph.env_states.obstacle
        keep = jnp.arange(n_dense) < n_keep
        if env_id == "CrazyFlie":
            c = jnp.where(keep[:, None], obs.center, jnp.full((3,), far, obs.center.dtype)[None])
            new = jax.vmap(Sphere.create)(c, obs.radius)
        else:
            c = jnp.where(keep[:, None], obs.center, jnp.full((2,), far, obs.center.dtype)[None])
            new = jax.vmap(Rectangle.create)(c, obs.width, obs.height, obs.theta)
        es = graph.env_states
        o0 = env._get_obs(es.agent, es.goal, new)
        return env.get_graph(es._replace(obstacle=new, obs_history=jnp.repeat(o0[None], env.history_len, 0),
                                         action_history=jnp.zeros_like(es.action_history)))

    return jax.jit(jax.vmap(one))(graphs)


def summarize(unsafe, finish, seeds, epi):
    """Per-seed mean, sample std and pooled Wilson CI of safe, reach and success."""
    out = {}
    for name, ok in (("safe", 1 - unsafe), ("reach", finish), ("success", (1 - unsafe) * finish)):
        per_seed = np.array([ok[j * epi:(j + 1) * epi].mean() for j in range(len(seeds))])
        n, p, z = ok.size, ok.mean(), 1.96
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        out[name] = (100 * per_seed.mean(), 100 * (per_seed.std(ddof=1) if len(seeds) > 1 else np.nan),
                     100 * (c - h), 100 * (c + h))
    out["collisions"] = int(unsafe.sum())
    out["timeouts"] = int(((1 - unsafe) * (1 - finish)).sum())
    out["failures"] = [f"s{seeds[i // epi]}e{i % epi}:{'C' if unsafe[i] else 'T'}"
                       for i in range(len(unsafe)) if unsafe[i] or not finish[i]]
    return out


def fmt(m):
    """Format one metric as mean ± std [CI]."""
    return f"{m[0]:6.2f} ± {m[1]:5.2f} [{m[2]:5.1f},{m[3]:6.1f}]"


def print_table(title, cols, rows, show_failures, csv_path):
    """Print the results table."""
    head = cols + ["Safe %", "Reach %", "Success %", "Coll.", "T/O"]
    lines = [[*r[0], fmt(r[1]["safe"]), fmt(r[1]["reach"]), fmt(r[1]["success"]),
              str(r[1]["collisions"]), str(r[1]["timeouts"])] for r in rows]
    w = [max(len(h), *(len(l[i]) for l in lines)) for i, h in enumerate(head)]
    print("\n" + title)
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
    print("  ".join("-" * x for x in w))
    for l in lines:
        print("  ".join(v.ljust(w[i]) for i, v in enumerate(l)))
    print("Values: mean ± sample std (ddof=1) over evaluation seeds [pooled Wilson 95% CI]. "
          "Coll. = episodes with a collision; T/O = safe episodes that did not reach the goal.")
    if show_failures:
        for r in rows:
            print(f"  failures {' '.join(r[0])}: {' '.join(r[1]['failures']) or 'none'}")
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(cols + [f"{m}_{s}" for m in ("safe", "reach", "success") for s in ("mean", "std", "ci_lo", "ci_hi")]
                        + ["collisions", "timeouts"])
            for r in rows:
                wr.writerow(list(r[0]) + [f"{x:.4f}" for m in ("safe", "reach", "success") for x in r[1][m]]
                            + [r[1]["collisions"], r[1]["timeouts"]])
        print(f"CSV: {csv_path}")


def main(args):
    """Run the selected test mode."""
    env_id = ENVS.get(args.env, args.env)
    cfg = load_config(args.ckpt)
    ck = pickle.load(open(os.path.join(args.ckpt, "models", str(args.step), "actor.pkl"), "rb"))
    arch = infer_architecture(ck["params"] if isinstance(ck, dict) and "params" in ck else ck)
    K, horizon = arch["history_len"], int(cfg.get("horizon", 6))
    area_train, n_train = float(cfg.get("area_size", 4.0)), int(cfg.get("n_obs", 8))
    keys = episode_keys(args.seeds, args.epi)

    olr_normal = args.obs_len_range if args.obs_len_range is not None else (
        [0.2, 0.7] if env_id == "CrazyFlie" and args.mode == "normal" else None)
    env = make_env(env_id, cfg, K, area_train, n_train, args.max_step or 256, olr_normal)
    if env.obs_dim != arch["obs_dim"]:
        raise ValueError(f"env obs_dim {env.obs_dim} != checkpoint {arch['obs_dim']}")
    algo = build_algo(env, cfg, arch, horizon)
    algo.load(os.path.join(args.ckpt, "models"), args.step)

    if args.mode == "normal":
        unsafe, finish, reward, cost = run(env, algo, horizon, reset_graphs(env, keys), keys)
        print_episodes(unsafe, finish, reward, cost, args.seeds, args.epi)
        res = summarize(unsafe, finish, args.seeds, args.epi)
        print_table(f"NORMAL TEST  (l={area_train:g}, M={n_train}, {env.max_episode_steps}-step episodes)",
                    ["Env", "Model"], [((env_id, os.path.basename(args.ckpt.rstrip("/"))), res)],
                    args.show_failures, args.csv)

    elif args.mode == "generalization":
        triples = ([tuple(float(x) if i == 0 else int(x) for i, x in enumerate(t.split(","))) for t in args.triples.split()]
                   if args.triples else DEFAULT_TRIPLES[env_id])
        olr = args.obs_len_range if args.obs_len_range is not None else (
            [0.1, 0.6] if env_id == "CrazyFlie" else None)
        npos = 3 if env_id == "CrazyFlie" else 2
        rows = []
        for l, nd, nm, ns in triples:
            genv = make_env(env_id, cfg, K, l, nd, args.max_step or 1024, olr)
            dense = reset_graphs(genv, keys)
            for n in (nd, nm, ns):
                unsafe, finish, _, _ = run(genv, algo, horizon, remove_obstacles(env_id, genv, dense, nd, n, l), keys)
                res = summarize(unsafe, finish, args.seeds, args.epi)
                rows.append(((f"{l:g}", str(n), f"{n / l ** npos:.2f}"), res))
                print(f"  l={l:g} M={n}: success {fmt(res['success'])}", flush=True)
        print_table(f"GENERALIZATION  ({env_id}, {args.max_step or 1024}-step episodes, obstacle size "
                    f"{genv._params['obs_len_range']})", ["l", "M", "rho"], rows, args.show_failures, args.csv)

    else:
        if env_id != "DoubleIntegrator":
            raise SystemExit("--mode robustness is implemented for DoubleIntegrator (DI) only")
        from barrierformer.env.double_integrator import DoubleIntegrator

        class PerturbedDI(DoubleIntegrator):
            """DoubleIntegrator with velocity mismatch and additive process noise."""
            def __init__(self, *a, alpha_vel=1.0, sigma_vec=None, **kw):
                """Store the perturbation settings."""
                super().__init__(*a, **kw)
                self._alpha_vel = float(alpha_vel)
                self._sigma = np.zeros(4, np.float32) if sigma_vec is None else np.asarray(sigma_vec, np.float32)
                self._noise_key = None

            def agent_step_euler(self, s, u):
                """Perturbed simulator step."""
                x_dot = self.agent_xdot(s, u)
                if self._alpha_vel != 1.0:
                    x_dot = x_dot.at[:, :2].multiply(self._alpha_vel)
                nxt = x_dot * self.dt + s
                if np.any(self._sigma > 0):
                    nxt = nxt + jr.normal(self._noise_key, nxt.shape) * jnp.asarray(self._sigma)[None]
                return self.clip_state(nxt)

        v_max = float(np.asarray(env.state_lim()[1])[2])
        scale = (np.array([area_train, area_train, v_max, v_max], np.float32) if args.noise_scale == "state"
                 else np.ones(4, np.float32))
        rows = []
        for kind, vals in (("p_dot = alpha_vel*v", ALPHA_VEL), (f"x_dot = f + w ({args.noise_scale})", SIGMAS)):
            for v in vals:
                kw = dict(alpha_vel=v) if kind.startswith("p_dot") else dict(sigma_vec=v * scale)
                penv = make_env(env_id, cfg, K, area_train, n_train, args.max_step or 256, args.obs_len_range,
                                cls=PerturbedDI, **kw)
                unsafe, finish, _, _ = run(penv, algo, horizon, reset_graphs(penv, keys), keys)
                res = summarize(unsafe, finish, args.seeds, args.epi)
                rows.append(((kind, f"{v:g}"), res))
                print(f"  {kind} {v:g}: success {fmt(res['success'])}", flush=True)
        print_table(f"ROBUSTNESS  (DI, noise sigma_vec = sigma * {scale.tolist()})", ["Perturbation", "Value"],
                    rows, args.show_failures, args.csv)


if __name__ == "__main__":
    main(ARGS)

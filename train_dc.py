"""Fine-tune the BarrierFormer policy on DubinsCar."""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import argparse
import datetime
import pickle
import numpy as np
import wandb
import yaml

from barrierformer.env.dubins_car import DubinsCar
from barrierformer.algo.gcbf_plus import GCBFPlus
from barrierformer.trainer.trainer import Trainer
from barrierformer.trainer.utils import is_connected


def assert_params_match(model_params, ckpt_params, ckpt_path: str) -> None:
    """Fail loudly when a checkpoint's geometry disagrees with the CLI config."""
    import jax.tree_util as jtu

    def leaves(tree):
        return {jtu.keystr(k): np.shape(v)
                for k, v in jtu.tree_flatten_with_path(tree)[0]}

    want, got = leaves(model_params), leaves(ckpt_params)
    problems = []
    for k in sorted(set(want) | set(got)):
        if k not in got:
            problems.append(f"    {k}: missing from checkpoint (model wants {want[k]})")
        elif k not in want:
            problems.append(f"    {k}: extra in checkpoint (shape {got[k]})")
        elif want[k] != got[k]:
            problems.append(f"    {k}: model {want[k]} vs checkpoint {got[k]}")

    if problems:
        raise ValueError(
            f"Checkpoint geometry does not match the CLI configuration.\n"
            f"  checkpoint: {ckpt_path}\n"
            f"  {len(problems)} mismatched parameter(s):\n"
            + "\n".join(problems[:20])
            + ("\n    ..." if len(problems) > 20 else "")
            + "\n  Fix the architecture flags (--hidden-dim, --num-heads, "
              "--num-layers, --mlp-ratio,\n  --action-head-hidden-dim, "
              "--action-head-num-layers, --dynamics-head-hidden-dim,\n  "
              "--dynamics-head-num-layers, --history-len) to match the checkpoint."
        )


def check_pretrain_meta(ckpt_path: str, expected: dict, state_obs: str = "raw", pid_k_v: float = 2.3) -> None:
    """Compare a pretrained checkpoint's params_*.meta.json with this run's settings."""
    import json
    meta_path = os.path.splitext(ckpt_path)[0] + ".meta.json"
    if not os.path.exists(meta_path):
        print(f"  [meta] WARNING: no {os.path.basename(meta_path)} next to the checkpoint — "
              f"cannot verify LiDAR order / history length it was pretrained with")
        return
    with open(meta_path) as f:
        meta = json.load(f)
    ckpt_state_obs = meta.get('state_obs', 'raw')
    ckpt_kv = float(meta.get('pid_k_v') or 2.3)
    if abs(ckpt_kv - pid_k_v) > 1e-9:
        print(f"  [meta] NOTE: action head pretrained with pid_k_v={ckpt_kv}, this run uses --pid-k-v {pid_k_v} "
              f"(dynamics head / observation unaffected; the action head adapts during fine-tuning)")
    if ckpt_state_obs != state_obs:
        raise ValueError(f"{ckpt_path} was pretrained with state_obs={ckpt_state_obs!r} but this run "
                         f"uses --state-obs {state_obs}")
    problems = [f"    {k}: checkpoint={meta[k]!r}  this run={v!r}"
                for k, v in expected.items() if k in meta and meta[k] != v]
    if problems:
        raise ValueError("Pretrained checkpoint was trained with different settings:\n"
                         + "\n".join(problems) + f"\n  meta: {meta_path}")
    print(f"  [meta] pretraining settings match: "
          + ", ".join(f"{k}={meta.get(k)!r}" for k in expected) + f", state_obs={ckpt_state_obs!r}, pid_k_v={ckpt_kv}")


def train(args):
    """Build the environments and algorithm, then run training."""
    print(f"> Running train_transformer_dubins.py")
    print(f"  horizon={args.horizon}  history_len={args.history_len}  "
          f"n_agents={args.num_agents}  area={args.area_size}  steps={args.steps}")

    if not is_connected():
        os.environ["WANDB_MODE"] = "offline"
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"

    np.random.seed(args.seed)

    env_params = {**DubinsCar.PARAMS, "n_obs": args.n_obs, "n_rays": args.n_rays,
                  "safe_buffer_mult": args.safe_buffer_mult,
                  "lidar_ray_order": args.lidar_ray_order,
                  "use_orig_u_ref": args.use_orig_u_ref,
                  "state_obs": args.state_obs,
                  "pid_k_v": args.pid_k_v}
    print(f"  nominal PID: {'ORIGINAL gcbf+' if args.use_orig_u_ref else 'alt'}  "
          f"lidar_ray_order={args.lidar_ray_order}  state_obs={args.state_obs}  max_step={args.max_step}")
    env = DubinsCar(
        num_agents=args.num_agents,
        area_size=args.area_size,
        max_step=args.max_step,
        history_len=args.history_len,
        params=env_params,
    )
    env_test = DubinsCar(
        num_agents=args.num_agents,
        area_size=args.area_size,
        max_step=args.max_step,
        history_len=args.history_len,
        params=env_params,
    )

    print(f"  env.obs_dim={env.obs_dim}  state_dim={env.state_dim}  "
          f"action_dim={env.action_dim}  history_len={env.history_len}")

    algo = GCBFPlus(
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=args.num_agents,
        gnn_layers=args.gnn_layers,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        horizon=args.horizon,
        lr_actor=args.lr_actor,
        lr_cbf=args.lr_cbf,
        alpha=args.alpha,
        eps=0.02,
        inner_epoch=args.inner_epoch,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        loss_action_coef=args.loss_action_coef,
        loss_unsafe_coef=args.loss_unsafe_coef,
        loss_safe_coef=args.loss_safe_coef,
        loss_dt_cbf_coef=args.loss_dt_cbf_coef,
        loss_dyn_coef=args.loss_dyn_coef,
        sqp_horizon_len=args.horizon,
        n_sqp_iter=args.n_sqp_iter,
        relax_penalty=args.relax_penalty,
        sqp_skip_safe=args.sqp_skip_safe,
        sqp_exit_tol=args.sqp_exit_tol,
        sqp_dedup=args.sqp_dedup,
        simulator_rollout=args.simulator_rollout,
        labeling_horizon=args.labeling_horizon,
        beta=args.beta,
        gamma=args.gamma,
        cbf_tgt_tau=args.cbf_tgt_tau,
        total_steps=args.steps if args.lr_schedule else None,
        tf_hidden_dim=args.hidden_dim,
        tf_num_heads=args.num_heads,
        tf_num_layers=args.num_layers,
        tf_mlp_ratio=args.mlp_ratio,
        tf_dropout=args.dropout,
        action_head_hidden_dim=args.action_head_hidden_dim,
        action_head_num_layers=args.action_head_num_layers,
        dynamics_head_hidden_dim=args.dynamics_head_hidden_dim,
        dynamics_head_num_layers=args.dynamics_head_num_layers,
        use_robust_phase2_training=args.use_robust_phase2_training,
        robust_phase2_alpha=args.robust_phase2_alpha,
        sqp_diagnostics=args.sqp_diagnostics,
        sqp_filter_nonconverged=args.sqp_filter_nonconverged,
        sqp_filter_violating=args.sqp_filter_violating,
        sqp_diag_csv=args.sqp_diag_csv,
        sqp_diag_env_name=args.sqp_diag_env_name,
        sqp_dynamics=args.sqp_dynamics,
        sqp_learned_jac=args.sqp_learned_jac,
        sqp_learned_after=args.sqp_learned_after,
    )

    if args.pretrain_ckpt:
        with open(args.pretrain_ckpt, 'rb') as f:
            pretrained_params = pickle.load(f)
        if isinstance(pretrained_params, dict) and 'params' in pretrained_params \
                and 'opt_state' in pretrained_params:
            pretrained_params = pretrained_params['params']
        assert_params_match(algo.tf_actor_train_state.params, pretrained_params,
                            args.pretrain_ckpt)
        check_pretrain_meta(args.pretrain_ckpt, {'lidar_ray_order': args.lidar_ray_order, 'history_len': args.history_len, 'use_orig_u_ref': args.use_orig_u_ref},
                            state_obs=args.state_obs, pid_k_v=args.pid_k_v)
        algo.tf_actor_train_state = algo.tf_actor_train_state.replace(
            params=pretrained_params)
        print(f"  Loaded pretrained transformer weights from {args.pretrain_ckpt}")

    tf_params  = sum(np.array(p).size for p in __import__('jax').tree_util.tree_leaves(algo.tf_actor_train_state.params))
    cbf_params = sum(np.array(p).size for p in __import__('jax').tree_util.tree_leaves(algo.cbf_mlp_train_state.params))
    print(f"  Transformer params : {tf_params:,}")
    print(f"  CBFMLP params      : {cbf_params:,}")

    start_time  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir     = os.path.join(args.log_dir, "DubinsCar", "gcbf_transformer", f"seed{args.seed}_{start_time}")
    run_name    = (f"gcbf_tf_DC_H{args.horizon}_hist{args.history_len}_{start_time}" if args.name is None else args.name)

    os.makedirs(log_dir, exist_ok=True)

    train_params = {
        "run_name":       run_name,
        "training_steps": args.steps,
        "eval_interval":  args.eval_interval,
        "eval_epi":       args.eval_epi,
        "save_interval":  args.save_interval,
    }

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        n_env_train=args.n_env_train,
        n_env_test=args.n_env_test,
        seed=args.seed,
        params=train_params,
        save_log=not args.debug,)

    wandb.config.update(vars(args))
    wandb.config.update(algo.config)
    if not args.debug:
        cfg_path = os.path.join(log_dir, "config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(vars(args),    f, default_flow_style=False)
            yaml.dump(algo.config,   f, default_flow_style=False)
        print(f"  Config saved to {cfg_path}")

    if args.load_dir and args.load_step is not None:
        algo.load(args.load_dir, args.load_step)
        print(f"  Resumed from {args.load_dir} step {args.load_step}")

    print(f"  Log dir : {log_dir}")
    print(f"  Run name: {run_name}")
    print(f"\n  Starting training for {args.steps} steps ...\n")

    trainer.train()

    if args.sqp_diagnostics:
        csv_path = algo.write_sqp_diagnostics()
        algo.print_sqp_diagnostics_summary()
        if csv_path:
            print(f"\n  SQP diagnostics written to {csv_path} "
                  f"({len(algo.sqp_diag_rows)} rows)")


def main():
    """Parse the command-line arguments and start training."""
    parser = argparse.ArgumentParser(
        description="Train transformer CBF (GCBFPlus) on DubinsCar"
    )

    parser.add_argument("-n", "--num-agents",  type=int,   default=1,
                        help="Number of agents (default: 1)")
    parser.add_argument("--area-size",         type=float, required=True,
                        help="Side length of the square world (metres)")
    parser.add_argument("--max-step",          type=int,   default=256,
                        help="Max episode length (default: 256, as in the pretraining data and the "
                             "DoubleIntegrator / CrazyFlie runs)")
    parser.add_argument("--n-rays",            type=int,   default=32,
                        help="LiDAR rays per agent (default: 32)")
    parser.add_argument("--n-obs",             type=int,   default=8,
                        help="Number of obstacles per episode (default: 8)")
    parser.add_argument("--safe-buffer-mult",  type=float, default=1.0,
                        help="CBF label keep-out dilation (default 1.0 = original behavior). "
                             ">1.0 enlarges the obstacle radius in safe/unsafe masks so the "
                             "barrier brakes earlier; does NOT change the true collision cost. "
                             "Use for the braking-distance ablation (e.g. 1.5, 2.0).")
    parser.add_argument("--lidar-ray-order",   action="store_true", default=False,
                        help="Store LiDAR in ray order (slot i = ray i) instead of the original "
                             "distance-sorted order, so each hit is paired with its own ray angle "
                             "and obstacle direction is observable. Changes the observation: the "
                             "--pretrain-ckpt must have been pretrained on data collected with the "
                             "same setting (collector --lidar-ray-order). Default OFF = original.")
    parser.add_argument("--state-obs",         type=str, default="raw", choices=["raw", "vxy"],
                        help="Agent-state part of the observation (transformer + CBF input): "
                             "raw = [x, y, theta, v] (original; theta unbounded), vxy = "
                             "[x, y, v cos(theta), v sin(theta)]. Env state, physics, SQP and the "
                             "dynamics-head target stay raw. Must match the checkpoint's meta.json.")
    parser.add_argument("--pid-k-v",           type=float, default=2.3,
                        help="Speed gain k_v of the original gcbf+ PID (default 2.3 = original). Must "
                             "match the checkpoint's meta.json pid_k_v (pretraining --relabel-pid-kv).")
    parser.add_argument("--use-orig-u-ref",    action=argparse.BooleanOptionalAction, default=True,
                        help="Nominal controller u_nom = the ORIGINAL gcbf+ PID (default ON; matches "
                             "the sorted-ray/DC data, collected with --use-orig-u-ref, and DI, which "
                             "uses the original gcbf+ LQR). --no-use-orig-u-ref = the alt PID. Must "
                             "match the PID the pretraining data was labelled against.")

    parser.add_argument("--horizon",           type=int,   default=18,
                        help="Predictive rollout horizon H (default: 18)")
    parser.add_argument("--history-len",       type=int,   default=32,
                        help="Transformer context window T+1 (default: 32). "
                             "MUST match the pretrained checkpoint (sets pos_embed = 2*(history_len-1)+1).")
    parser.add_argument("--hidden-dim",              type=int,   default=128,
                        help="Transformer hidden width d_model (default: 128)")
    parser.add_argument("--num-heads",               type=int,   default=2,
                        help="Attention heads (default: 2)")
    parser.add_argument("--num-layers",              type=int,   default=1,
                        help="Transformer blocks (default: 1)")
    parser.add_argument("--mlp-ratio",               type=float, default=4.0,
                        help="Transformer FFN width multiplier (default: 4.0)")
    parser.add_argument("--dropout",                 type=float, default=0.0,
                        help="Transformer dropout rate (default: 0.0)")
    parser.add_argument("--action-head-hidden-dim",  type=int,   default=128,
                        help="Action head hidden width (default: 128)")
    parser.add_argument("--action-head-num-layers",  type=int,   default=4,
                        help="Action head hidden layers (default: 4)")
    parser.add_argument("--dynamics-head-hidden-dim",type=int,   default=128,
                        help="Dynamics head hidden width (default: 128)")
    parser.add_argument("--dynamics-head-num-layers",type=int,   default=3,
                        help="Dynamics head hidden layers (default: 3)")

    parser.add_argument("--n-sqp-iter",        type=int,   default=25,
                        help="SQP iterations per transition (default: 25)")
    parser.add_argument("--sqp-skip-safe",     action=argparse.BooleanOptionalAction, default=False,
                        help="Run the SQP iteration loop only on samples whose nominal plan "
                             "violates the CBF condition. Faster, but NOT label-identical: a "
                             "different batch shape changes the floating-point path and the SQP "
                             "can land on a different solution (measured: ~24%% of labels differ). "
                             "Default OFF.")
    parser.add_argument("--sqp-exit-tol",      type=float, default=0.0,
                        help="Stop a sample's SQP once its largest DTCBF residual is <= this "
                             "(default 0.0 = original exact test). > 0 changes labels slightly.")
    parser.add_argument("--sqp-dedup",         action="store_true", default=False,
                        help="Solve each distinct SQP input once and copy the result to its "
                             "duplicates. NOT label-identical (same reason as --sqp-skip-safe); "
                             "small speed-up. Default OFF.")
    parser.add_argument("--simulator-rollout", type=lambda v: str(v).lower() in ("1", "true", "yes"),
                        default=False, nargs="?", const=True,
                        help="ABLATION (default False): the rollout given to the CBF losses (single-step "
                             "and H-step horizon) is produced by the simulator (real env.step physics) "
                             "instead of the transformer dynamics head. Use --simulator-rollout=True.")
    parser.add_argument("--relax-penalty",     type=float, default=1e3,
                        help="Slack penalty in SQP QP objective (default: 1000)")

    parser.add_argument("--sqp-dynamics", type=str, default="true",
                        choices=["true", "learned"],
                        help="Transition model the SQP teacher integrates over "
                             "its horizon: 'true' = env.step_simulator "
                             "(default; byte-identical to the pre-ablation "
                             "teacher), 'learned' = the transformer dynamics "
                             "head with the context window rolled "
                             "autoregressively. Replaces the transition model "
                             "ONLY — u_nom and observation reconstruction stay "
                             "analytic.")
    parser.add_argument("--sqp-learned-jac", type=str, default="full",
                        choices=["full", "state-only"],
                        help="Jacobian scope for --sqp-dynamics learned. "
                             "'full' (default) differentiates dc/dU through "
                             "both the state chain and the context-window "
                             "updates; 'state-only' stop_gradients the context "
                             "window so z_k is constant within an SQP "
                             "iteration — cheaper, approximate Jacobian. "
                             "No effect when --sqp-dynamics true.")
    parser.add_argument("--sqp-learned-after", type=int, default=0,
                        help="Training step at which the teacher switches from "
                             "the analytic to the learned transition model "
                             "(default: 0 = immediately). No effect when "
                             "--sqp-dynamics true.")

    parser.add_argument("--sqp-diagnostics", action="store_true", default=False,
                        help="Collect per-solve SQP convergence/slack diagnostics "
                             "and write results/sqp_stats.csv (default: OFF). "
                             "Observation only — does NOT change label selection.")
    parser.add_argument("--sqp-filter-nonconverged", action="store_true", default=False,
                        help="Exclude samples whose SQP solve did not strictly "
                             "converge (max c > 0) from L_act. Requires "
                             "--sqp-diagnostics. Default OFF = existing behaviour "
                             "(the 0.05-residual gate alone).")
    parser.add_argument("--sqp-filter-violating", action="store_true", default=False,
                        help="Exclude samples whose SQP solution violates the "
                             "UNTIGHTENED DTCBF condition from L_act. Requires "
                             "--sqp-diagnostics. Default OFF.")
    parser.add_argument("--sqp-diag-csv", type=str, default="results/sqp_stats.csv",
                        help="Path for the SQP diagnostics CSV "
                             "(default: results/sqp_stats.csv)")
    parser.add_argument("--sqp-diag-env-name", type=str, default="DubinsCar",
                        help="Environment name written to the diagnostics CSV "
                             "(default: DubinsCar)")

    parser.add_argument("--steps",             type=int,   default=1000,
                        help="Training steps / episodes (default: 1000)")
    parser.add_argument("--batch-size",        type=int,   default=256,
                        help="Minibatch size per inner update (default: 256)")
    parser.add_argument("--buffer-size",       type=int,   default=512,
                        help="Replay buffer capacity in episodes (default: 512)")
    parser.add_argument("--inner-epoch",       type=int,   default=8,
                        help="Inner gradient steps per rollout (default: 8)")
    parser.add_argument("--gnn-layers",        type=int,   default=1,
                        help="GNN layers (legacy param, keep 1)")
    parser.add_argument("--seed",              type=int,   default=0)

    parser.add_argument("--lr-actor",          type=float, default=3e-5)
    parser.add_argument("--lr-cbf",            type=float, default=3e-5)
    parser.add_argument("--alpha",             type=float, default=1.0,
                        help="DTCBF decay rate α (default: 1.0)")
    parser.add_argument("--max-grad-norm",     type=float, default=2.0,
                        help="Global gradient clip norm for CBF and actor (default: 2.0)")

    parser.add_argument("--loss-action-coef",   type=float, default=0.1,
                        help="Actor imitation loss weight (default: 0.1)")
    parser.add_argument("--loss-unsafe-coef",   type=float, default=1.0)
    parser.add_argument("--loss-safe-coef",     type=float, default=1.0)
    parser.add_argument("--loss-dt-cbf-coef",   type=float, default=0.2,
                        help="Horizon log-sum-exp DTCBF loss weight (default: 0.2)")
    parser.add_argument("--loss-dyn-coef",      type=float, default=1.0,
                        help="Dynamics head supervised loss weight (default: 1.0)")
    parser.add_argument("--beta",               type=float, default=10.0,
                        help="Log-sum-exp sharpness β (default: 10.0)")
    parser.add_argument("--gamma",              type=float, default=0.0,
                        help="SQP DTCBF offset γ (default: 0.0)")
    parser.add_argument("--cbf-tgt-tau",        type=float, default=0.5,
                        help="Polyak coefficient for CBF target network update: "
                             "tgt = tau*train + (1-tau)*tgt (default: 0.5)")

    parser.add_argument("--use-robust-phase2-training", action="store_true", default=False,
                        help="Enable robustness training: inject zero-mean Gaussian "
                             "disturbance into the Phase-2 predictive rollout's "
                             "dynamics-head next-state (default: OFF).")
    parser.add_argument("--robust-phase2-alpha",  type=float, default=0.075,
                        help="Std-dev (scale) alpha of the Phase-2 robustness "
                             "disturbance (default: 0.075). Only used when "
                             "--use-robust-phase2-training is set.")

    parser.add_argument("--labeling-horizon",    type=int,   default=32,
                        help="Steps before collision marked unlabeled (default: 32)")
    parser.add_argument("--n-env-train",        type=int,   default=16,
                        help="Parallel envs for training rollouts (default: 16)")
    parser.add_argument("--n-env-test",         type=int,   default=32,
                        help="Parallel envs for evaluation (default: 32)")
    parser.add_argument("--log-dir",            type=str,   default="./logs")
    parser.add_argument("--eval-interval",      type=int,   default=1)
    parser.add_argument("--eval-epi",           type=int,   default=5)
    parser.add_argument("--save-interval",      type=int,   default=10)
    parser.add_argument("--name",               type=str,   default=None,
                        help="WandB run name override")
    parser.add_argument("--lr-schedule",        action="store_true", default=False,
                        help="Enable cosine LR decay from lr_max to 0.1*lr_max over training")
    parser.add_argument("--debug",              action="store_true", default=False,
                        help="Disable JIT and WandB for quick debugging")
    parser.add_argument("--deterministic",      action="store_true", default=False,
                        help="Enable bit-for-bit reproducible GPU ops (xla_gpu_deterministic_ops, "
                             "TF_CUDNN_DETERMINISTIC, TF_DETERMINISTIC_OPS). Off by default; "
                             "slightly slower but removes run-to-run nondeterminism.")
    parser.add_argument("--load-dir",           type=str,   default=None,
                        help="Path to models/ dir of a previous run to resume from")
    parser.add_argument("--load-step",          type=int,   default=None,
                        help="Checkpoint step number to load (e.g. 10)")
    parser.add_argument("--pretrain-ckpt",      type=str,   default=None,
                        help="Path to a pretrained transformer params_*.pkl to warm-start from")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

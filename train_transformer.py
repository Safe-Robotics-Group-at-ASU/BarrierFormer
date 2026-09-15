"""
train_transformer.py
====================
Training script for the Transformer-based CBF (BarrierFormer) on DoubleIntegrator.

Differences from the original train.py (GNN-based):
  - Uses CausalTransformerPolicy instead of GNN actor
  - Transformer-specific hyperparameters: history_len, sqp_horizon, n_sqp_iter,
    loss_dt_cbf_coef, loss_dyn_coef, beta, gamma
  - history_len passed directly to DoubleIntegrator (not in make_env)
  - rollout_transformer used instead of rollout

Default configuration:
  - horizon (predictive rollout H) = 6
  - history_len (context window T+1) = 12
  - environment: DoubleIntegrator, 1 agent
  - batch_size = 256
  - inner_epoch = 8

Usage:
    python train_transformer.py --area-size 4.0
    python train_transformer.py --area-size 4.0 --num-agents 1 --steps 2000 --debug
    python train_transformer.py --area-size 4.0 --n-env-train 16 --steps 5000 --name my_run
"""

import argparse
import datetime
import os
import pickle
import numpy as np
import wandb
import yaml

from barrierformer.env.double_integrator import DoubleIntegrator
from barrierformer.algo.barrierformer import BarrierFormer
from barrierformer.trainer.trainer import Trainer
from barrierformer.trainer.utils import is_connected


def train(args):
    print(f"> Running train_transformer.py")
    print(f"  horizon={args.horizon}  history_len={args.history_len}  "
          f"n_agents={args.num_agents}  area={args.area_size}  steps={args.steps}")

    # ── Environment variables ────────────────────────────────────────────────
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
    if not is_connected():
        os.environ["WANDB_MODE"] = "offline"
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"

    np.random.seed(args.seed)

    # ── Environments ─────────────────────────────────────────────────────────
    # DoubleIntegrator is instantiated directly so that history_len can be set. make_env() does not expose history_len.
    # Merge CLI overrides into the default PARAMS so unchanged keys keep their defaults.
    env_params = {**DoubleIntegrator.PARAMS, "n_obs": args.n_obs, "n_rays": args.n_rays}
    env = DoubleIntegrator(
        num_agents=args.num_agents,
        area_size=args.area_size,
        max_step=args.max_step,
        history_len=args.history_len,
        params=env_params,
    )
    env_test = DoubleIntegrator(
        num_agents=args.num_agents,
        area_size=args.area_size,
        max_step=args.max_step,
        history_len=args.history_len,
        params=env_params,
    )

    print(f"  env.obs_dim={env.obs_dim}  state_dim={env.state_dim}  "
          f"action_dim={env.action_dim}  history_len={env.history_len}")

    # ── Algorithm ────────────────────────────────────────────────────────────
    algo = BarrierFormer(
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
        max_grad_norm=2.0,
        seed=args.seed,
        loss_action_coef=args.loss_action_coef,
        loss_unsafe_coef=args.loss_unsafe_coef,
        loss_safe_coef=args.loss_safe_coef,
        loss_dt_cbf_coef=args.loss_dt_cbf_coef,
        loss_dyn_coef=args.loss_dyn_coef,
        sqp_horizon_len=args.horizon,   # SQP horizon matches transformer rollout horizon
        n_sqp_iter=args.n_sqp_iter,
        relax_penalty=args.relax_penalty,
        labeling_horizon=args.labeling_horizon,
        beta=args.beta,
        gamma=args.gamma,
        total_steps=args.steps if args.lr_schedule else None,
        dyn_scale_path=args.dyn_scale_path,
        freeze_backbone=args.freeze_backbone,
    )

    # Load pretrained transformer weights if a checkpoint path was provided.
    if args.pretrain_ckpt:
        with open(args.pretrain_ckpt, 'rb') as f:
            pretrained_params = pickle.load(f)
        algo.tf_actor_train_state = algo.tf_actor_train_state.replace(
            params=pretrained_params)
        print(f"  Loaded pretrained transformer weights from {args.pretrain_ckpt}")

    tf_params  = sum(np.array(p).size for p in __import__('jax').tree_util.tree_leaves(algo.tf_actor_train_state.params))
    cbf_params = sum(np.array(p).size for p in __import__('jax').tree_util.tree_leaves(algo.cbf_mlp_train_state.params))
    print(f"  Transformer params : {tf_params:,}")
    print(f"  CBFMLP params      : {cbf_params:,}")

    # ── Logging directories ──────────────────────────────────────────────────
    start_time  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir     = os.path.join(args.log_dir, "DoubleIntegrator", "gcbf_transformer", f"seed{args.seed}_{start_time}")
    run_name    = (f"gcbf_tf_DI_H{args.horizon}_hist{args.history_len}_{start_time}" if args.name is None else args.name)

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


def main():
    parser = argparse.ArgumentParser(
        description="Train transformer CBF (BarrierFormer) on DoubleIntegrator"
    )

    # ── Environment ──────────────────────────────────────────────────────────
    parser.add_argument("-n", "--num-agents",  type=int,   default=1,
                        help="Number of agents (default: 1)")
    parser.add_argument("--area-size",         type=float, required=True,
                        help="Side length of the square world (metres)")
    parser.add_argument("--max-step",          type=int,   default=256,
                        help="Max episode length (default: 256)")
    parser.add_argument("--n-rays",            type=int,   default=32,
                        help="LiDAR rays per agent (default: 32)")
    parser.add_argument("--n-obs",             type=int,   default=8,
                        help="Number of obstacles per episode (default: 8)")

    # ── Transformer architecture ──────────────────────────────────────────────
    parser.add_argument("--horizon",           type=int,   default=6,
                        help="Predictive rollout horizon H (default: 6)")
    parser.add_argument("--history-len",       type=int,   default=12,
                        help="Transformer context window T+1 (default: 12)")

    # ── SQP teacher ──────────────────────────────────────────────────────────
    parser.add_argument("--n-sqp-iter",        type=int,   default=25,
                        help="SQP iterations per transition (default: 25)")
    parser.add_argument("--relax-penalty",     type=float, default=1e3,
                        help="Slack penalty in SQP QP objective (default: 1000)")

    # ── Training loop ────────────────────────────────────────────────────────
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

    # ── Optimisation ─────────────────────────────────────────────────────────
    parser.add_argument("--lr-actor",          type=float, default=3e-5)
    parser.add_argument("--lr-cbf",            type=float, default=3e-5)
    parser.add_argument("--alpha",             type=float, default=1.0,
                        help="DTCBF decay rate α (default: 1.0)")

    # ── Loss coefficients ────────────────────────────────────────────────────
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

    # ── Logging / evaluation ─────────────────────────────────────────────────
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
    parser.add_argument("--load-dir",           type=str,   default=None,
                        help="Path to models/ dir of a previous run to resume from")
    parser.add_argument("--load-step",          type=int,   default=None,
                        help="Checkpoint step number to load (e.g. 10)")
    parser.add_argument("--pretrain-ckpt",      type=str,   default=None,
                        help="Path to a pretrained transformer params_*.pkl to warm-start from")
    parser.add_argument("--dyn-scale-path",     type=str,   default=None,
                        help="Path to a .npz with a 'scale' array (shape [state_dim]). Use when the "
                             "pretrained dynamics head predicts the NORMALISED increment dx/scale "
                             "instead of raw dx (required for CrazyFlie). Omit for DI/DubinsCar, "
                             "whose heads output raw deltas (unit scale, exact no-op).")
    parser.add_argument("--freeze-backbone",    action="store_true", default=False,
                        help="Frozen-backbone ablation (paper Table 9): train ONLY the actor head in "
                             "Phase 2, keeping the Phase-1 pretrained transformer and dynamics head "
                             "fixed.")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

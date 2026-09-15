from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import wandb
from flax.training import train_state

# ── Import transformer from CBF-TF (this file lives inside CBF-TF/) ──────────
sys.path.insert(0, str(Path(__file__).parent))
from barrierformer.nn.model import CausalTransformerPolicy, TransformerPolicyConfig
from barrierformer.trainer.utils import is_connected


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_chunk(path: str, obs_version: int) -> dict[str, np.ndarray]:
    """
    Load one .npz chunk and flatten (N_epi, T_max, ...) → (N_total, ...).
    Selects obs_seq or obs_seq_v2 based on obs_version, drops the other.
    """
    raw      = np.load(path)
    N_epi, T_max = raw['obs_seq'].shape[:2]
    obs_key  = 'obs_seq_v2' if obs_version == 2 else 'obs_seq'
    drop_key = 'obs_seq'    if obs_version == 2 else 'obs_seq_v2'
    flat = {}
    for k in raw.files:
        if k == drop_key:
            continue
        arr = raw[k]
        flat[k] = arr.reshape(N_epi * T_max, *arr.shape[2:])
    if obs_key != 'obs_seq':
        flat['obs_seq'] = flat.pop(obs_key)
    return flat


def get_chunk_files(data_path: str) -> list:
    """
    Accept either a directory of chunk_*.npz files or a single .npz file.
    Returns a sorted list of Path objects.
    """
    p = Path(data_path)
    if p.is_dir():
        chunks = sorted(p.glob("chunk_*.npz"))
        if not chunks:
            raise FileNotFoundError(f"No chunk_*.npz files found in {p}")
        return chunks
    return [p]   # single file treated as one chunk


def split_chunks(chunk_files: list, val_frac: float) -> tuple:
    """Split chunk list into train/val at chunk level — no data is copied."""
    n_val = max(1, int(val_frac * len(chunk_files)))
    return chunk_files[:-n_val], chunk_files[-n_val:]


def iter_batches(
    dataset: dict[str, np.ndarray],
    batch_size: int,
    rng: np.random.Generator,
):
    """Yield shuffled mini-batches from a single loaded chunk."""
    N   = len(dataset['obs_seq'])
    idx = rng.permutation(N)
    for start in range(0, N - batch_size + 1, batch_size):
        b = idx[start: start + batch_size]
        yield {k: dataset[k][b] for k in dataset}


# ─────────────────────────────────────────────────────────────────────────────
# Model + TrainState
# ─────────────────────────────────────────────────────────────────────────────

def build_state(args, obs_dim: int, act_dim: int, state_dim: int,
                history_len: int, key: jnp.ndarray) -> train_state.TrainState:
    """Initialise CausalTransformerPolicy and wrap in a Flax TrainState."""
    cfg = TransformerPolicyConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        obs_dim=obs_dim,
        action_dim=act_dim,
        state_dim=state_dim,
        max_seq_len=2 * (history_len - 1) + 1,   # 2*11+1 = 23 for history_len=12
        mlp_ratio=4.0,
        dropout_rate=args.dropout,
        activation="gelu",
        action_head_hidden_dim=64,
        action_head_num_layers=2,
        action_head_activation="tanh",
        dynamics_head_hidden_dim=128,
        dynamics_head_num_layers=3,
        dynamics_head_activation="gelu",
    )
    model = CausalTransformerPolicy(config=cfg)

    # Init with deterministic=True (no dropout RNG needed)
    dummy_obs = jnp.zeros((1, history_len,       obs_dim))
    dummy_act = jnp.zeros((1, history_len - 1,   act_dim))
    dummy_nom = jnp.zeros((1,                    act_dim))
    params = model.init(key, dummy_obs, dummy_act, dummy_nom, deterministic=True)

    optimizer = optax.chain(
        optax.clip_by_global_norm(2.0),
        optax.adamw(learning_rate=args.lr, weight_decay=1e-3),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


# ─────────────────────────────────────────────────────────────────────────────
# JIT-compiled train / eval steps
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def train_step(
    state: train_state.TrainState,
    obs_seq:       jnp.ndarray,   # (B, HISTORY_LEN,   obs_dim)
    act_seq:       jnp.ndarray,   # (B, HISTORY_LEN-1, act_dim)
    u_nom:         jnp.ndarray,   # (B, act_dim)
    delta_u_label: jnp.ndarray,   # (B, act_dim)   — action head target
    delta_x_label: jnp.ndarray,   # (B, state_dim) — dynamics head target
    loss_action_coef: float,
    loss_dyn_coef:    float,
    dropout_key:   jnp.ndarray,
) -> tuple[train_state.TrainState, dict]:

    def _loss(params):
        out = state.apply_fn(
            params, obs_seq, act_seq, u_nom,
            deterministic=False,
            rngs={'dropout': dropout_key},
        )
        # out.delta_u:    (B, act_dim)
        # out.delta_x_hat:(B, state_dim)
        loss_action = jnp.mean(jnp.sum((out.delta_u     - delta_u_label) ** 2, axis=-1))
        loss_dyn    = jnp.mean(jnp.sum((out.delta_x_hat - delta_x_label) ** 2, axis=-1))
        loss_total  = loss_action_coef * loss_action + loss_dyn_coef * loss_dyn
        return loss_total, {
            'train/loss_action': loss_action,
            'train/loss_dyn':    loss_dyn,
            'train/loss_total':  loss_total,
        }

    (_, info), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, info


@jax.jit
def eval_step(
    state: train_state.TrainState,
    obs_seq:       jnp.ndarray,
    act_seq:       jnp.ndarray,
    u_nom:         jnp.ndarray,
    delta_u_label: jnp.ndarray,
    delta_x_label: jnp.ndarray,
    loss_action_coef: float,
    loss_dyn_coef:    float,
) -> dict:
    out = state.apply_fn(
        state.params, obs_seq, act_seq, u_nom, deterministic=True,
    )
    loss_action = jnp.mean(jnp.sum((out.delta_u     - delta_u_label) ** 2, axis=-1))
    loss_dyn    = jnp.mean(jnp.sum((out.delta_x_hat - delta_x_label) ** 2, axis=-1))
    loss_total  = loss_action_coef * loss_action + loss_dyn_coef * loss_dyn
    return {
        'val/loss_action': loss_action,
        'val/loss_dyn':    loss_dyn,
        'val/loss_total':  loss_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch prediction printer
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def predict(state: train_state.TrainState,
            obs_seq: jnp.ndarray,
            act_seq: jnp.ndarray,
            u_nom:   jnp.ndarray):
    """Deterministic forward pass — returns (delta_u, delta_x_hat)."""
    out = state.apply_fn(state.params, obs_seq, act_seq, u_nom, deterministic=True)
    return out.delta_u, out.delta_x_hat


def print_batch_predictions(
    state:     train_state.TrainState,
    batch:     dict[str, np.ndarray],
    label_key: str,
    epoch:     int,
    n_samples: int = 5,
) -> None:
    """Print inputs, predictions and labels for the first n_samples of a batch.

    Called once per epoch on the first mini-batch so you can watch the model
    improve across epochs.

    For each sample prints:
      INPUT
        obs_seq  shape + current obs token (state, rel_goal, lidar norm)
                       + oldest  obs token (context start)
        act_seq  shape + most-recent action + oldest action in window
        u_nom    nominal LQR action
      ACTOR HEAD
        label    delta_u_label (actor / QP correction)
        pred     delta_u from transformer
        error    pred − label
      DYNAMICS HEAD
        label    delta_x_label (true state increment)
        pred     delta_x_hat from transformer
        error    pred − label
    """
    B = batch['obs_seq'].shape[0]
    n = min(n_samples, B)

    du_pred, dx_pred = predict(
        state,
        jnp.array(batch['obs_seq'][:n]),
        jnp.array(batch['act_seq'][:n]),
        jnp.array(batch['u_nom'][:n]),
    )
    du_pred = np.array(du_pred)   # (n, 2)
    dx_pred = np.array(dx_pred)   # (n, 4)

    bar = "─" * 68
    print(f"\n{'='*68}")
    print(f"  Epoch {epoch}  |  batch_size={B}  |  showing first {n} samples")
    print(f"{'='*68}")

    for i in range(n):
        obs_win  = batch['obs_seq'][i]       # (HISTORY_LEN, obs_dim)
        act_win  = batch['act_seq'][i]       # (HISTORY_LEN-1, act_dim)
        u_nom    = batch['u_nom'][i]         # (2,)
        du_label = batch[label_key][i]       # (2,)
        dx_label = batch['delta_x_label'][i] # (4,)

        # Parse current obs token: [state(4) | rel_goal(2) | lidar(64)]
        cur_obs  = obs_win[-1]               # most recent token
        old_obs  = obs_win[0]               # oldest token in window
        cur_act  = act_win[-1]              # most recent action in window
        old_act  = act_win[0]              # oldest action in window

        print(f"\n  ── Sample {i} {bar[:50]}")

        # ── INPUT ─────────────────────────────────────────────────────────
        obs_dim = obs_win.shape[-1]   # 70 (Way 1) or 134 (Way 2)

        print(f"  INPUT")
        def fmt_obs(obs_token: np.ndarray, label: str) -> None:
            state    = obs_token[:4]
            rel_goal = obs_token[4:6]
            lidar    = obs_token[6:]
            print(f"      {label}")
            print(f"        state    (x,y,vx,vy) : [{state[0]:+.4f} {state[1]:+.4f} "
                  f"{state[2]:+.4f} {state[3]:+.4f}]")
            print(f"        rel_goal (rx,ry)      : [{rel_goal[0]:+.4f} {rel_goal[1]:+.4f}]")
            if obs_dim == 70:
                # Way 1: lidar = (64,) = 32 rays × (rel_x, rel_y)
                rays = lidar.reshape(32, 2)
                print(f"        lidar    (Way 1 — 32 rays, rel endpoint (x,y)):")
                for row in range(0, 32, 4):
                    chunk = rays[row:row + 4]
                    pairs = "  ".join(
                        f"ray{row+j:02d}=({chunk[j,0]:+.3f},{chunk[j,1]:+.3f})"
                        for j in range(len(chunk)))
                    print(f"          {pairs}")
            else:
                # Way 2: lidar = (128,) = 32 rays × (hit_mask, dist_norm, cos_θ, sin_θ)
                rays = lidar.reshape(32, 4)
                print(f"        lidar    (Way 2 — 32 rays, hit|dist_norm|cos|sin):")
                for row in range(0, 32, 4):
                    chunk = rays[row:row + 4]
                    pairs = "  ".join(
                        f"ray{row+j:02d}=[h={chunk[j,0]:.0f} d={chunk[j,1]:.3f}"
                        f" c={chunk[j,2]:+.3f} s={chunk[j,3]:+.3f}]"
                        for j in range(len(chunk)))
                    print(f"          {pairs}")

        print(f"    obs_seq  shape={list(obs_win.shape)}  "
              f"(HISTORY_LEN={obs_win.shape[0]}, obs_dim={obs_win.shape[1]})")
        fmt_obs(cur_obs, f"current obs  token (t)   — all {obs_dim} values:")
        fmt_obs(old_obs, f"oldest  obs  token (t-{len(obs_win)-1}) — all {obs_dim} values:")
        print(f"    act_seq  shape={list(act_win.shape)}")
        print(f"      most recent action (t-1) : [{cur_act[0]:+.4f} {cur_act[1]:+.4f}]")
        print(f"      oldest  action    (t-{len(act_win)}) : [{old_act[0]:+.4f} {old_act[1]:+.4f}]")
        print(f"    u_nom (LQR nominal)        : [{u_nom[0]:+.4f} {u_nom[1]:+.4f}]")

        # ── ACTOR HEAD ────────────────────────────────────────────────────
        du_err = du_pred[i] - du_label
        print(f"  ACTOR HEAD  (delta_u = u_applied − u_nom)")
        print(f"    label ('{label_key}') : [{du_label[0]:+.6f}  {du_label[1]:+.6f}]")
        print(f"    prediction             : [{du_pred[i][0]:+.6f}  {du_pred[i][1]:+.6f}]")
        print(f"    error  (pred − label)  : [{du_err[0]:+.6f}  {du_err[1]:+.6f}]  "
              f"‖e‖={np.linalg.norm(du_err):.6f}")

        # ── DYNAMICS HEAD ─────────────────────────────────────────────────
        dx_err = dx_pred[i] - dx_label
        print(f"  DYNAMICS HEAD  (delta_x = x_{{t+1}} − x_t)")
        print(f"    label (delta_x_label)  : "
              f"[{dx_label[0]:+.6f}  {dx_label[1]:+.6f}  {dx_label[2]:+.6f}  {dx_label[3]:+.6f}]")
        print(f"    prediction             : "
              f"[{dx_pred[i][0]:+.6f}  {dx_pred[i][1]:+.6f}  {dx_pred[i][2]:+.6f}  {dx_pred[i][3]:+.6f}]")
        print(f"    error  (pred − label)  : "
              f"[{dx_err[0]:+.6f}  {dx_err[1]:+.6f}  {dx_err[2]:+.6f}  {dx_err[3]:+.6f}]  "
              f"‖e‖={np.linalg.norm(dx_err):.6f}")

    print(f"\n{'='*68}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(params, ckpt_dir: Path, tag: str) -> Path:
    path = ckpt_dir / f"params_{tag}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(params, f)
    return path


def load_checkpoint(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

    key = jr.PRNGKey(args.seed)
    rng = np.random.default_rng(args.seed)

    # ── Dataset — chunk-aware ─────────────────────────────────────────────────
    chunk_files            = get_chunk_files(args.data)
    trn_chunks, val_chunks = split_chunks(chunk_files, val_frac=0.1)

    # Peek at first chunk to get dims (no full load needed)
    _peek     = load_chunk(str(trn_chunks[0]), args.obs_version)
    _, HISTORY_LEN, obs_dim = _peek['obs_seq'].shape
    act_dim   = _peek['u_nom'].shape[-1]
    state_dim = _peek['delta_x_label'].shape[-1]
    del _peek

    label_key = 'delta_u_qp' if args.use_qp_label else 'delta_u_label'

    # ── Model ─────────────────────────────────────────────────────────────────
    key, init_key = jr.split(key)
    state = build_state(args, obs_dim, act_dim, state_dim, HISTORY_LEN, init_key)

    # Load pretrained checkpoint if requested
    if args.load_ckpt:
        state = state.replace(params=load_checkpoint(args.load_ckpt))

    # ── WandB — mirrors train.py / Trainer pattern exactly ───────────────────
    if not is_connected():
        os.environ["WANDB_MODE"] = "offline"
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
    import datetime
    run_name = (f"pretrain_obs{args.obs_version}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                if args.name is None else args.name)
    wandb.login()
    wandb.init(name=run_name, project=args.wandb_project, dir=args.ckpt_dir)
    wandb.config.update(vars(args))

    # ── Checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Scalar coefs as JAX arrays so JIT doesn't retrace on Python float changes
    la_coef  = jnp.float32(args.loss_action_coef)
    ld_coef  = jnp.float32(args.loss_dyn_coef)

    global_step = 0
    best_val    = float('inf')

    for epoch in range(1, args.epochs + 1):

        # ── Train — loop over all training chunks ──────────────────────────────
        ep_losses: dict[str, list] = {
            'train/loss_action': [],
            'train/loss_dyn':    [],
            'train/loss_total':  [],
        }

        trn_order = rng.permutation(len(trn_chunks))   # shuffle chunk order each epoch
        for ci in trn_order:
            chunk = load_chunk(str(trn_chunks[ci]), args.obs_version)
            for batch in iter_batches(chunk, args.batch_size, rng):
                key, drop_key = jr.split(key)
                state, info = train_step(
                    state,
                    jnp.array(batch['obs_seq']),
                    jnp.array(batch['act_seq']),
                    jnp.array(batch['u_nom']),
                    jnp.array(batch[label_key]),
                    jnp.array(batch['delta_x_label']),
                    la_coef,
                    ld_coef,
                    drop_key,
                )
                log = {k: float(v) for k, v in info.items()}
                wandb.log(log, step=global_step)
                for k in ep_losses:
                    ep_losses[k].append(log[k])
                global_step += 1

        # ── Validation — loop over all val chunks ──────────────────────────────
        val_losses: dict[str, list] = {
            'val/loss_action': [],
            'val/loss_dyn':    [],
            'val/loss_total':  [],
        }
        for val_chunk_path in val_chunks:
            val_chunk = load_chunk(str(val_chunk_path), args.obs_version)
            n_val_samples = min(args.batch_size * 8, len(val_chunk['obs_seq']))
            val_idx  = rng.choice(len(val_chunk['obs_seq']), size=n_val_samples, replace=False)
            val_info = eval_step(
                state,
                jnp.array(val_chunk['obs_seq'][val_idx]),
                jnp.array(val_chunk['act_seq'][val_idx]),
                jnp.array(val_chunk['u_nom'][val_idx]),
                jnp.array(val_chunk[label_key][val_idx]),
                jnp.array(val_chunk['delta_x_label'][val_idx]),
                la_coef,
                ld_coef,
            )
            for k, v in val_info.items():
                val_losses[k].append(float(v))
        val_log = {k: float(np.mean(v)) for k, v in val_losses.items()}
        wandb.log(val_log, step=global_step)

        # ── Console summary ───────────────────────────────────────────────────
        trn_total  = np.mean(ep_losses['train/loss_total'])
        trn_action = np.mean(ep_losses['train/loss_action'])
        trn_dyn    = np.mean(ep_losses['train/loss_dyn'])
        val_total  = val_log['val/loss_total']

        # Epoch-level summary — smooth curves in WandB (one point per epoch)
        wandb.log({
            'epoch/train_loss_total':  trn_total,
            'epoch/train_loss_action': trn_action,
            'epoch/train_loss_dyn':    trn_dyn,
            'epoch/val_loss_total':    val_log['val/loss_total'],
            'epoch/val_loss_action':   val_log['val/loss_action'],
            'epoch/val_loss_dyn':      val_log['val/loss_dyn'],
            'epoch':                   epoch,
        }, step=global_step)

        print(
            f"Epoch {epoch:>4}/{args.epochs}"
            f"  train_total={trn_total:.5f}"
            f"  (action={trn_action:.5f}  dyn={trn_dyn:.5f})"
            f"  val_total={val_total:.5f}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % args.save_every == 0:
            save_checkpoint(state.params, ckpt_dir, f"epoch{epoch:04d}")

        if val_total < best_val:
            best_val = val_total
            save_checkpoint(state.params, ckpt_dir, "best")

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint(state.params, ckpt_dir, "final")

    wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Open-loop pretraining of CausalTransformerPolicy"
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--data", type=str, required=True,
                        help="Path to pretrain_dataset.npz (or debug_dataset.npz)")
    parser.add_argument("--obs-version", type=int, default=2, choices=[1, 2],
                        help="Observation encoding: 1=raw lidar xy (70-dim), "
                             "2=4-feature lidar hit/dist/cos/sin (134-dim) [default: 2]")
    parser.add_argument("--use-qp-label", action="store_true", default=False,
                        help="Use QP correction as action-head label (default: actor correction)")

    # ── Architecture — must match train_transformer.py ────────────────────────
    parser.add_argument("--hidden-dim",  type=int,   default=128)
    parser.add_argument("--num-heads",   type=int,   default=2)
    parser.add_argument("--num-layers",  type=int,   default=1)
    parser.add_argument("--dropout",     type=float, default=0.1)

    # ── Optimisation ──────────────────────────────────────────────────────────
    parser.add_argument("--epochs",            type=int,   default=100)
    parser.add_argument("--batch-size",        type=int,   default=256)
    parser.add_argument("--lr",                type=float, default=3e-4)
    parser.add_argument("--loss-action-coef",  type=float, default=1.0,
                        help="Weight for action-head MSE loss (default: 1.0)")
    parser.add_argument("--loss-dyn-coef",     type=float, default=1.0,
                        help="Weight for dynamics-head MSE loss (default: 1.0)")
    parser.add_argument("--seed",              type=int,   default=0)

    # ── Logging / checkpointing ───────────────────────────────────────────────
    parser.add_argument("--wandb-project", type=str, default="cbf-tf-pretrain")
    parser.add_argument("--name",          type=str, default=None,
                        help="WandB run name (auto-generated if omitted)")
    parser.add_argument("--ckpt-dir",      type=str, default="./pretrain_ckpts",
                        help="Directory to save checkpoints")
    parser.add_argument("--save-every",    type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--load-ckpt",     type=str, default=None,
                        help="Path to a params_*.pkl to resume from")
    parser.add_argument("--debug",         action="store_true", default=False,
                        help="Disable WandB and use debug_dataset.npz settings")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

"""
Top-level CausalTransformerPolicy module.

Wires together the causal transformer backbone, the residual action head,
and the dynamics prediction head into a single Flax module.

Single-step forward pass (used at every real timestep):
    1.  (all_hidden, z_t) = transformer(obs_seq, action_seq)
    2.  delta_u           = action_head(z_t)
    3.  u_applied         = u_nom + delta_u
    4.  delta_x_hat       = dynamics_head(z_t, u_applied)

H-step autoregressive rollout (inference only):
    - Iterates single-step forward, slides the history window, and collects
      predicted states, observations, and actions.
    - Implemented as a plain Python loop so the caller can optionally jit it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from .transformer import CausalTransformer, TransformerConfig
from .action_head import ActionHead
from .dynamics_head import DynamicsHead


# Output containers (plain Python dataclasses, NOT Flax structs)

@dataclass
class ModelOutput:
    """Output of one CausalTransformerPolicy forward pass.

    Fields
    ------
    z_t               : Transformer last-token latent.  Shape: (batch, hidden_dim).
    delta_u           : Residual action correction.     Shape: (batch, action_dim).
    u_applied         : Final applied action (u_nom + delta_u).
                        Shape: (batch, action_dim).
    delta_x_hat       : Predicted state increment.      Shape: (batch, state_dim).
    all_hidden_states : Full transformer hidden sequence.
                        Shape: (batch, 2*T+1, hidden_dim).
    """

    z_t: jnp.ndarray
    delta_u: jnp.ndarray
    u_applied: jnp.ndarray
    delta_x_hat: jnp.ndarray
    all_hidden_states: jnp.ndarray


@dataclass
class RolloutOutput:
    """Output of an H-step autoregressive rollout.

    Fields
    ------
    x_hat_seq : Predicted states at each rollout step.
                Shape: (batch, H, state_dim).
    o_hat_seq : Predicted observations at each rollout step.
                Shape: (batch, H, obs_dim).
    u_seq     : Applied actions at each rollout step.
                Shape: (batch, H, action_dim).
    """

    x_hat_seq: jnp.ndarray
    o_hat_seq: jnp.ndarray
    u_seq: jnp.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TransformerPolicyConfig:
    """Aggregated configuration for the full CausalTransformerPolicy.

    All sub-config fields are derived from this single object inside the
    CausalTransformerPolicy.setup() method.

    Transformer backbone
    --------------------
    hidden_dim      : Model width; must be divisible by num_heads.
    num_heads       : Number of attention heads.
    num_layers      : Number of stacked transformer blocks.
    mlp_ratio       : FFN inner-width multiplier (default 4.0).
    dropout_rate    : Dropout probability (default 0.1).
    max_seq_len     : Maximum sequence length = 2*T+1.
    activation      : Activation for transformer FFN ('gelu', 'relu', 'tanh').

    Problem dimensions
    ------------------
    obs_dim         : Dimensionality of a single observation o_t.
    action_dim      : Dimensionality of a single action u_t.
    state_dim       : Dimensionality of the raw state x_t (for dynamics head).

    Action head
    -----------
    action_head_hidden_dim   : Width of hidden layers in the action head.
    action_head_num_layers   : Number of hidden layers in the action head.
    action_head_activation   : Activation for action head ('tanh', 'relu', ...).

    Dynamics head
    -------------
    dynamics_head_hidden_dim : Width of hidden layers in the dynamics head.
    dynamics_head_num_layers : Number of hidden layers in the dynamics head.
    dynamics_head_activation : Activation for dynamics head ('gelu', 'relu', ...).
    """

    # Transformer backbone
    hidden_dim: int
    num_heads: int
    num_layers: int
    obs_dim: int
    action_dim: int
    state_dim: int
    max_seq_len: int
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    activation: str = "gelu"

    # Action head
    action_head_hidden_dim: int = 64
    action_head_num_layers: int = 2
    action_head_activation: str = "tanh"

    # Dynamics head
    dynamics_head_hidden_dim: int = 128
    dynamics_head_num_layers: int = 3
    dynamics_head_activation: str = "gelu"


# Top-level policy module

class CausalTransformerPolicy(nn.Module):
    """
    Full transformer-based policy: transformer + action head + dynamics head.
    Inputs at each real timestep:
        obs_seq    : (batch, T+1, obs_dim)  — history window + current obs
        action_seq : (batch, T, action_dim) — T past actions
        u_nom      : (batch, action_dim)    — nominal action from controller

    The model produces:
        u_applied  = u_nom + delta_u        (action to execute)
        delta_x_hat                         (predicted state increment)

    Attributes
    ----------
    config : TransformerPolicyConfig with all hyperparameters.
    """

    config: TransformerPolicyConfig

    def setup(self) -> None:
        cfg = self.config

        tf_cfg = TransformerConfig(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            mlp_ratio=cfg.mlp_ratio,
            dropout_rate=cfg.dropout_rate,
            max_seq_len=cfg.max_seq_len,
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            activation=cfg.activation,
        )
        self.transformer = CausalTransformer(config=tf_cfg)

        self.action_head = ActionHead(
            hidden_dim=cfg.hidden_dim,
            action_dim=cfg.action_dim,
            head_hidden_dim=cfg.action_head_hidden_dim,
            num_layers=cfg.action_head_num_layers,
            activation=cfg.action_head_activation,
        )

        self.dynamics_head = DynamicsHead(
            hidden_dim=cfg.hidden_dim,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            head_hidden_dim=cfg.dynamics_head_hidden_dim,
            num_layers=cfg.dynamics_head_num_layers,
            activation=cfg.dynamics_head_activation,
        )

    def __call__(
        self,
        obs_seq: jnp.ndarray,      # (batch, T+1, obs_dim)
        action_seq: jnp.ndarray,   # (batch, T, action_dim)
        u_nom: jnp.ndarray,        # (batch, action_dim)
        deterministic: bool = True,
    ) -> ModelOutput:
        """Single-step forward pass through the full policy.

        Parameters
        ----------
        obs_seq       : Observation history (T past) + current obs.
                        Shape: (batch, T+1, obs_dim).
        action_seq    : Past action history (T actions).
                        Shape: (batch, T, action_dim).
        u_nom         : Nominal action from the external controller (e.g. LQR).
                        Shape: (batch, action_dim).
        deterministic : If False, enables dropout in the transformer.

        Returns
        -------
        ModelOutput with fields z_t, delta_u, u_applied, delta_x_hat,
        all_hidden_states.
        """
        # ── 1. Encode history window ─────────────────────────────────────────
        all_hidden, z_t = self.transformer(obs_seq, action_seq, deterministic)
        # all_hidden : (batch, 2*T+1, hidden_dim)
        # z_t        : (batch, hidden_dim)

        # ── 2. Residual action correction ────────────────────────────────────
        delta_u   = self.action_head(z_t)        # (batch, action_dim)
        u_applied = u_nom + delta_u              # (batch, action_dim)

        # ── 3. Dynamics prediction ───────────────────────────────────────────
        delta_x_hat = self.dynamics_head(z_t, u_applied)  # (batch, state_dim)

        return ModelOutput(
            z_t=z_t,
            delta_u=delta_u,
            u_applied=u_applied,
            delta_x_hat=delta_x_hat,
            all_hidden_states=all_hidden,
        )

    def call_dynamics_head(
        self,
        z_t: jnp.ndarray,         # (batch, hidden_dim)
        u_applied: jnp.ndarray,   # (batch, action_dim)
    ) -> jnp.ndarray:
        """Direct entry point for the dynamics head.

        Lets the caller feed a detached (z_t, u_applied) when computing
        ``loss_dyn``, so that L_dyn updates only the dynamics-head parameters
        (paper Sec. 4.6), and lets the SQP teacher / horizon rollout query the
        head with their OWN candidate action rather than the actor's.

        Invoked via ``policy.apply(params, z, u, method=...)``; the algo layer
        depends on it, so it must exist on the module.
        """
        return self.dynamics_head(z_t, u_applied)

    def rollout(
        self,
        params: dict,
        obs_seq: jnp.ndarray,          # (batch, T+1, obs_dim)
        action_seq: jnp.ndarray,       # (batch, T, action_dim)
        u_nom_seq: jnp.ndarray,        # (batch, H, action_dim)
        x_current: jnp.ndarray,        # (batch, state_dim) — initial state
        obs_from_state_fn: Callable,   # x_hat: (batch, state_dim) -> o_hat: (batch, obs_dim)
        action_lb: Optional[jnp.ndarray] = None,  # (action_dim,) clip lower bound
        action_ub: Optional[jnp.ndarray] = None,  # (action_dim,) clip upper bound
        deterministic: bool = True,
    ) -> RolloutOutput:
        """H-step autoregressive rollout (inference only, no parameter updates).

        At each step k the model predicts:
          - delta_u   via the action head
          - delta_x_hat via the dynamics head

        Then the history window is slid forward by appending the new
        (observation, action) pair and dropping the oldest.

        This method is NOT JIT-compiled internally. Wrap with jax.jit at the
        call site if needed. Note that ``obs_from_state_fn`` must itself be
        JIT-compatible if you JIT the outer call.

        Parameters
        ----------
        params           : Frozen Flax parameter dict (from model.init(...)['params']).
        obs_seq          : Initial observation window (T past + current obs).
                           Shape: (batch, T+1, obs_dim).
        action_seq       : Initial action window (T past actions).
                           Shape: (batch, T, action_dim).
        u_nom_seq        : Nominal actions for each rollout step.
                           Shape: (batch, H, action_dim).
        x_current        : Current known state (starting point for delta integration).
                           Shape: (batch, state_dim).
        obs_from_state_fn: Function mapping predicted state -> observation.
                           Signature: (batch, state_dim) -> (batch, obs_dim).
                           Implemented outside nn/ (e.g. in env/double_integrator.py).
        deterministic    : If False, dropout is active (not recommended for rollout).

        Returns
        -------
        RolloutOutput with:
            x_hat_seq : (batch, H, state_dim)   — predicted states
            o_hat_seq : (batch, H, obs_dim)      — predicted observations
            u_seq     : (batch, H, action_dim)   — applied actions
        """
        H = u_nom_seq.shape[1]

        x_hat_list = []
        o_hat_list = []
        u_list     = []

        x_hat = x_current  # running state estimate

        for k in range(H):
            u_nom_k = u_nom_seq[:, k, :]   # (batch, action_dim)

            # ── Single-step model forward ──────────────────────────────────
            out: ModelOutput = self.apply(
                params,
                obs_seq,
                action_seq,
                u_nom_k,
                deterministic,
            )

            # ── State update ───────────────────────────────────────────────
            x_hat = x_hat + out.delta_x_hat          # (batch, state_dim)
            o_hat = obs_from_state_fn(x_hat)          # (batch, obs_dim)

            x_hat_list.append(x_hat)
            o_hat_list.append(o_hat)
            u_list.append(out.u_applied)

            # ── Slide observation and action windows ──────────────────────
            # Drop the oldest obs (index 0) and append the new predicted obs
            obs_seq = jnp.concatenate(
                [obs_seq[:, 1:, :], o_hat[:, None, :]], axis=1
            )
            # Clip u_applied if bounds are provided before sliding into
            # action_seq.  env.step() stores clip(action) in action_history,
            # so the rollout context must use the same clipped value to avoid a
            # training/inference distribution shift in the action channel.
            u_slide = (
                jnp.clip(out.u_applied, action_lb, action_ub)
                if action_lb is not None
                else out.u_applied
            )
            action_seq = jnp.concatenate(
                [action_seq[:, 1:, :], u_slide[:, None, :]], axis=1
            )

        # Stack collected outputs: list of H tensors -> (batch, H, dim)
        x_hat_seq = jnp.stack(x_hat_list, axis=1)   # (batch, H, state_dim)
        o_hat_seq = jnp.stack(o_hat_list, axis=1)   # (batch, H, obs_dim)
        u_seq     = jnp.stack(u_list,     axis=1)   # (batch, H, action_dim)

        return RolloutOutput(
            x_hat_seq=x_hat_seq,
            o_hat_seq=o_hat_seq,
            u_seq=u_seq,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick sanity check: instantiate, init, and run a forward pass.

    Run with:
        python -m barrierformer.nn.model
    """
    import jax

    # ── Problem dimensions ───────────────────────────────────────────────────
    T          = 4          # history length (past steps)
    H          = 3          # rollout horizon
    batch      = 2
    obs_dim    = 70
    action_dim = 2
    state_dim  = 4          # (x, y, vx, vy) for double integrator

    # ── Config with intentionally small dims ────────────────────────────────
    cfg = TransformerPolicyConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        max_seq_len=2 * T + 1,       # = 9
        mlp_ratio=2.0,
        dropout_rate=0.0,
        action_head_hidden_dim=16,
        action_head_num_layers=2,
        dynamics_head_hidden_dim=32,
        dynamics_head_num_layers=2,
    )

    model = CausalTransformerPolicy(config=cfg)

    # ── Random inputs ────────────────────────────────────────────────────────
    key = jax.random.PRNGKey(0)
    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    obs_seq    = jax.random.normal(k1, (batch, T + 1, obs_dim))
    action_seq = jax.random.normal(k2, (batch, T,     action_dim))
    u_nom      = jax.random.normal(k3, (batch,         action_dim))
    u_nom_seq  = jax.random.normal(k4, (batch, H,      action_dim))
    x_current  = jax.random.normal(key, (batch, state_dim))

    # ── Initialise parameters ────────────────────────────────────────────────
    params = model.init(jax.random.PRNGKey(1), obs_seq, action_seq, u_nom)

    print("─" * 60)
    print("CausalTransformerPolicy smoke test")
    print("─" * 60)
    print(f"  T (history length)  : {T}")
    print(f"  H (rollout horizon) : {H}")
    print(f"  seq_len (2T+1)      : {2*T+1}")
    print(f"  batch               : {batch}")
    print(f"  obs_dim             : {obs_dim}")
    print(f"  action_dim          : {action_dim}")
    print(f"  state_dim           : {state_dim}")
    print(f"  hidden_dim          : {cfg.hidden_dim}")
    print()

    # ── Single forward pass ──────────────────────────────────────────────────
    out = model.apply(params, obs_seq, action_seq, u_nom)

    print("Single-step forward pass output shapes:")
    print(f"  z_t               : {out.z_t}")
    print(f"  delta_u           : {out.delta_u}")
    print(f"  u_applied         : {out.u_applied}")
    print(f"  delta_x_hat       : {out.delta_x_hat}")
    print(f"  all_hidden_states : {out.all_hidden_states}")
    print()

    # ── Autoregressive rollout ───────────────────────────────────────────────
    def dummy_obs_from_state(x_hat: jnp.ndarray) -> jnp.ndarray:
        """Placeholder: returns zero-padded observation from state."""
        return jnp.concatenate(
            [x_hat, jnp.zeros((x_hat.shape[0], obs_dim - state_dim))], axis=-1
        )

    rollout_out = model.rollout(
        params,
        obs_seq,
        action_seq,
        u_nom_seq,
        x_current,
        dummy_obs_from_state,
    )

    print(f"H-step rollout output shapes (H={H}):")
    print(f"  x_hat_seq : {rollout_out.x_hat_seq}")
    print(f"  o_hat_seq : {rollout_out.o_hat_seq}")
    print(f"  u_seq     : {rollout_out.u_seq}")
    print()
    print("Smoke test PASSED.")

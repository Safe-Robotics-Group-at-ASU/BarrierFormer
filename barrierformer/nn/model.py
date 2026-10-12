"""Top-level CausalTransformerPolicy module."""

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


@dataclass
class ModelOutput:
    """Output of one CausalTransformerPolicy forward pass."""

    z_t: jnp.ndarray
    delta_u: jnp.ndarray
    u_applied: jnp.ndarray
    delta_x_hat: jnp.ndarray
    all_hidden_states: jnp.ndarray


@dataclass
class RolloutOutput:
    """Output of an H-step autoregressive rollout."""

    x_hat_seq: jnp.ndarray
    o_hat_seq: jnp.ndarray
    u_seq: jnp.ndarray


@dataclass(frozen=True)
class TransformerPolicyConfig:
    """Aggregated configuration for the full CausalTransformerPolicy."""

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

    action_head_hidden_dim: int = 64
    action_head_num_layers: int = 2
    action_head_activation: str = "tanh"

    dynamics_head_hidden_dim: int = 128
    dynamics_head_num_layers: int = 3
    dynamics_head_activation: str = "gelu"

    action_lb: Optional[tuple] = None
    action_ub: Optional[tuple] = None


class CausalTransformerPolicy(nn.Module):
    """Full transformer-based policy: transformer + action head + dynamics head."""

    config: TransformerPolicyConfig

    def setup(self) -> None:
        """Create the transformer, action head and dynamics head."""
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
        obs_seq: jnp.ndarray,
        action_seq: jnp.ndarray,
        u_nom: jnp.ndarray,
        deterministic: bool = True,
    ) -> ModelOutput:
        """Single-step forward pass through the full policy."""
        all_hidden, z_t = self.transformer(obs_seq, action_seq, deterministic)

        delta_u   = self.action_head(z_t)
        u_applied = u_nom + delta_u

        cfg = self.config
        if cfg.action_lb is not None and cfg.action_ub is not None:
            u_for_dyn = jnp.clip(
                u_applied,
                jnp.asarray(cfg.action_lb),
                jnp.asarray(cfg.action_ub),
            )
        else:
            u_for_dyn = u_applied
        delta_x_hat = self.dynamics_head(z_t, u_for_dyn)

        return ModelOutput(
            z_t=z_t,
            delta_u=delta_u,
            u_applied=u_applied,
            delta_x_hat=delta_x_hat,
            all_hidden_states=all_hidden,
        )

    def call_dynamics_head(
        self,
        z_t: jnp.ndarray,
        u_applied: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predict the state increment from a latent and an action."""
        return self.dynamics_head(z_t, u_applied)

    def rollout(
        self,
        params: dict,
        obs_seq: jnp.ndarray,
        action_seq: jnp.ndarray,
        u_nom_seq: jnp.ndarray,
        x_current: jnp.ndarray,
        obs_from_state_fn: Callable,
        action_lb: Optional[jnp.ndarray] = None,
        action_ub: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> RolloutOutput:
        """H-step autoregressive rollout (inference only, no parameter updates)."""
        H = u_nom_seq.shape[1]

        x_hat_list = []
        o_hat_list = []
        u_list     = []

        x_hat = x_current

        for k in range(H):
            u_nom_k = u_nom_seq[:, k, :]

            out: ModelOutput = self.apply(
                params,
                obs_seq,
                action_seq,
                u_nom_k,
                deterministic,
            )

            x_hat = x_hat + out.delta_x_hat
            o_hat = obs_from_state_fn(x_hat)

            x_hat_list.append(x_hat)
            o_hat_list.append(o_hat)
            u_list.append(out.u_applied)

            obs_seq = jnp.concatenate(
                [obs_seq[:, 1:, :], o_hat[:, None, :]], axis=1
            )
            u_slide = (
                jnp.clip(out.u_applied, action_lb, action_ub)
                if action_lb is not None
                else out.u_applied
            )
            action_seq = jnp.concatenate(
                [action_seq[:, 1:, :], u_slide[:, None, :]], axis=1
            )

        x_hat_seq = jnp.stack(x_hat_list, axis=1)
        o_hat_seq = jnp.stack(o_hat_list, axis=1)
        u_seq     = jnp.stack(u_list,     axis=1)

        return RolloutOutput(
            x_hat_seq=x_hat_seq,
            o_hat_seq=o_hat_seq,
            u_seq=u_seq,
        )


if __name__ == "__main__":
    import jax

    T          = 4
    H          = 3
    batch      = 2
    obs_dim    = 70
    action_dim = 2
    state_dim  = 4

    cfg = TransformerPolicyConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        max_seq_len=2 * T + 1,
        mlp_ratio=2.0,
        dropout_rate=0.0,
        action_head_hidden_dim=16,
        action_head_num_layers=2,
        dynamics_head_hidden_dim=32,
        dynamics_head_num_layers=2,
    )

    model = CausalTransformerPolicy(config=cfg)

    key = jax.random.PRNGKey(0)
    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    obs_seq    = jax.random.normal(k1, (batch, T + 1, obs_dim))
    action_seq = jax.random.normal(k2, (batch, T,     action_dim))
    u_nom      = jax.random.normal(k3, (batch,         action_dim))
    u_nom_seq  = jax.random.normal(k4, (batch, H,      action_dim))
    x_current  = jax.random.normal(key, (batch, state_dim))

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

    out = model.apply(params, obs_seq, action_seq, u_nom)

    print("Single-step forward pass output shapes:")
    print(f"  z_t               : {out.z_t}")
    print(f"  delta_u           : {out.delta_u}")
    print(f"  u_applied         : {out.u_applied}")
    print(f"  delta_x_hat       : {out.delta_x_hat}")
    print(f"  all_hidden_states : {out.all_hidden_states}")
    print()

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

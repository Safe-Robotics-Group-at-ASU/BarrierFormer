import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import optax
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import einops as ei
import os
import pickle

from typing import Callable, Optional, Tuple, NamedTuple
from flax.training.train_state import TrainState
from jaxproxqp.jaxproxqp import JaxProxQP

from barrierformer.utils.typing import Action, Params, PRNGKey, Array, State
from barrierformer.utils.graph import GraphsTuple
from barrierformer.utils.utils import merge01, jax_vmap, mask2index, tree_merge
from barrierformer.trainer.data import Rollout
from barrierformer.trainer.buffer import MaskedReplayBuffer, MaskedTransitionBuffer
from barrierformer.trainer.utils import compute_norm_and_clip, jax2np, tree_copy, empty_grad_tx
from barrierformer.env.base import MultiAgentEnv
from barrierformer.algo.module.cbf import CBF
from barrierformer.algo.module.policy import DeterministicPolicy
from barrierformer.algo.rollout import make_obs_from_state_fn
from barrierformer.nn.model import CausalTransformerPolicy, TransformerPolicyConfig
from barrierformer.nn.cbf import CBFMLP
from barrierformer.algo import sqp_diagnostics as sqp_diag
from barrierformer.algo.sqp_diagnostics import (
    SQPSolveDiagnostics, SQP_SLACK_TOL, SQP_CONV_THRESH,
    QP_PRI_RES_THRESH, QP_DUA_RES_THRESH,
    EXIT_CONVERGED, EXIT_MAX_ITERS, EXIT_NONFINITE,
)
from .gcbf import GCBF


class Batch(NamedTuple):
    """One training batch: graphs, safe/unsafe masks and teacher action labels."""
    graph: GraphsTuple
    safe_mask: Array
    unsafe_mask: Array
    u_qp: Action
    sqp_converged: Optional[Array] = None


class GCBFPlus(GCBF):

    """BarrierFormer fine-tuning algorithm: transformer policy, CBF MLP and SQP teacher."""
    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            gnn_layers: int,
            batch_size: int,
            buffer_size: int,
            horizon: int = 32,
            lr_actor: float = 3e-5,
            lr_cbf: float = 3e-5,
            alpha: float = 1.0,
            eps: float = 0.02,
            inner_epoch: int = 8,
            loss_action_coef: float = 0.1,
            loss_unsafe_coef: float = 1.,
            loss_safe_coef: float = 1.,
            max_grad_norm: float = 2.,
            seed: int = 0,
            **kwargs
    ):
        """Build the transformer policy, CBF network, optimizers and replay buffers."""
        super(GCBF, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_cbf = lr_cbf
        self.alpha = alpha
        self.eps = eps
        self.inner_epoch = inner_epoch
        self.loss_action_coef = loss_action_coef
        self.loss_unsafe_coef = loss_unsafe_coef
        self.loss_safe_coef = loss_safe_coef
        self.gnn_layers = gnn_layers
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.horizon = horizon

        nominal_graph = GraphsTuple(
            nodes=jnp.zeros((n_agents, node_dim)),
            edges=jnp.zeros((n_agents, edge_dim)),
            states=jnp.zeros((n_agents, state_dim)),
            n_node=jnp.array(n_agents),
            n_edge=jnp.array(n_agents),
            senders=jnp.arange(n_agents),
            receivers=jnp.arange(n_agents),
            node_type=jnp.zeros((n_agents,)),
            env_states=jnp.zeros((n_agents,)),
        )
        self.nominal_graph = nominal_graph

        self.cbf = CBF(
            node_dim=node_dim,
            edge_dim=edge_dim,
            n_agents=n_agents,
            gnn_layers=gnn_layers
        )
        key = jr.PRNGKey(seed)
        cbf_key, key = jr.split(key)
        cbf_params = self.cbf.net.init(cbf_key, nominal_graph, self.n_agents)
        cbf_optim = optax.adamw(learning_rate=lr_cbf, weight_decay=1e-3)
        self.cbf_optim = optax.apply_if_finite(cbf_optim, 1_000_000)
        self.cbf_train_state = TrainState.create(
            apply_fn=self.cbf.get_cbf,
            params=cbf_params,
            tx=self.cbf_optim
        )
        self.cbf_tgt = TrainState.create(apply_fn=self.cbf.get_cbf, params=tree_copy(cbf_params), tx=empty_grad_tx())

        self.actor = DeterministicPolicy(
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )
        actor_key, key = jr.split(key)
        actor_params = self.actor.net.init(actor_key, nominal_graph, self.n_agents)
        actor_optim = optax.adamw(learning_rate=lr_actor, weight_decay=1e-3)
        self.actor_optim = optax.apply_if_finite(actor_optim, 1_000_000)
        self.actor_train_state = TrainState.create(
            apply_fn=self.actor.sample_action,
            params=actor_params,
            tx=self.actor_optim
        )

        history_len = env.history_len
        _u_lb, _u_ub = env.action_lim()
        tf_cfg = TransformerPolicyConfig(
            hidden_dim=kwargs.get('tf_hidden_dim', 128),
            num_heads=kwargs.get('tf_num_heads', 2),
            num_layers=kwargs.get('tf_num_layers', 1),
            obs_dim=env.obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            max_seq_len=2 * (history_len - 1) + 1,
            mlp_ratio=kwargs.get('tf_mlp_ratio', 4.0),
            dropout_rate=kwargs.get('tf_dropout', 0.0),
            activation="gelu",
            action_head_hidden_dim=kwargs.get('action_head_hidden_dim', 128),
            action_head_num_layers=kwargs.get('action_head_num_layers', 3),
            dynamics_head_hidden_dim=kwargs.get('dynamics_head_hidden_dim', 128),
            dynamics_head_num_layers=kwargs.get('dynamics_head_num_layers', 3),
            action_lb=tuple(float(v) for v in _u_lb),
            action_ub=tuple(float(v) for v in _u_ub),
        )
        self.tf_policy = CausalTransformerPolicy(config=tf_cfg)

        tf_key, key = jr.split(key)
        dummy_obs    = jnp.zeros((n_agents, history_len,     env.obs_dim))
        dummy_act    = jnp.zeros((n_agents, history_len - 1, action_dim))
        dummy_u_nom  = jnp.zeros((n_agents, action_dim))
        tf_actor_params = self.tf_policy.init(tf_key, dummy_obs, dummy_act, dummy_u_nom)

        total_steps = kwargs.get('total_steps', None)
        inner_epoch  = inner_epoch

        def _make_schedule(lr: float) -> optax.Schedule:
            if total_steps is not None:
                decay_steps = max(total_steps * inner_epoch, 1)
                return optax.cosine_decay_schedule(
                    init_value=lr, decay_steps=decay_steps, alpha=0.1
                )
            return lr

        tf_actor_optim = optax.adamw(learning_rate=_make_schedule(lr_actor), weight_decay=1e-3)
        self.tf_actor_optim = optax.apply_if_finite(tf_actor_optim, 1_000_000)
        self.tf_actor_train_state = TrainState.create(
            apply_fn=self.tf_policy.apply,
            params=tf_actor_params,
            tx=self.tf_actor_optim,
        )

        self.loss_dt_cbf_coef  = kwargs.get('loss_dt_cbf_coef', 0.2)
        self.loss_dyn_coef     = kwargs.get('loss_dyn_coef', 1.0)
        self.sqp_horizon_len   = kwargs.get('sqp_horizon_len', 5)
        self.n_sqp_iter        = kwargs.get('n_sqp_iter', 25)
        self.relax_penalty     = kwargs.get('relax_penalty', 1e3)

        self.labeling_horizon  = kwargs.get('labeling_horizon', 32)

        self.beta             = kwargs.get('beta', 10.0)

        self.cbf_tgt_tau      = kwargs.get('cbf_tgt_tau', 0.5)

        self.gamma            = kwargs.get('gamma', 0.0)
        self._sqp_slack_mean  = 0.0
        self._sqp_slack_max   = 0.0

        self.sqp_diagnostics         = kwargs.get('sqp_diagnostics', False)
        self.sqp_filter_nonconverged = kwargs.get('sqp_filter_nonconverged', False)
        self.sqp_filter_violating    = kwargs.get('sqp_filter_violating', False)
        self.sqp_diag_csv            = kwargs.get('sqp_diag_csv', 'results/sqp_stats.csv')
        self.sqp_diag_env_name       = kwargs.get('sqp_diag_env_name', 'DI')
        self._sqp_diag_rows: list    = []
        if (self.sqp_filter_nonconverged or self.sqp_filter_violating) \
                and not self.sqp_diagnostics:
            raise ValueError(
                "sqp_filter_nonconverged / sqp_filter_violating require "
                "sqp_diagnostics=True (the filters read fields that only the "
                "diagnostics path computes). Pass --sqp-diagnostics."
            )

        self.sqp_dynamics = kwargs.get('sqp_dynamics', 'true')
        if self.sqp_dynamics not in ('true', 'learned'):
            raise ValueError(
                f"sqp_dynamics must be 'true' or 'learned', got {self.sqp_dynamics!r}"
            )

        self.sqp_learned_jac = kwargs.get('sqp_learned_jac', 'full')
        if self.sqp_learned_jac not in ('full', 'state-only'):
            raise ValueError(
                f"sqp_learned_jac must be 'full' or 'state-only', "
                f"got {self.sqp_learned_jac!r}"
            )

        self.sqp_learned_after = int(kwargs.get('sqp_learned_after', 0))

        self.sqp_skip_safe = bool(kwargs.get('sqp_skip_safe', False))
        self.sqp_exit_tol = float(kwargs.get('sqp_exit_tol', 0.0))
        self.sqp_dedup = bool(kwargs.get('sqp_dedup', False))
        self.simulator_rollout = bool(kwargs.get('simulator_rollout', False))
        if self.sqp_exit_tol < 0.0:
            raise ValueError(f"sqp_exit_tol must be >= 0, got {self.sqp_exit_tol}")
        self._sqp_learned_announced = False

        self.use_robust_phase2_training = kwargs.get('use_robust_phase2_training', False)
        self.robust_phase2_alpha        = kwargs.get('robust_phase2_alpha', 0.075)

        self.cbf_mlp = CBFMLP(obs_dim=env.obs_dim)
        cbf_mlp_key, key = jr.split(key)
        cbf_mlp_params = self.cbf_mlp.init(
            cbf_mlp_key, jnp.zeros((n_agents, env.obs_dim))
        )
        cbf_mlp_optim = optax.adamw(learning_rate=_make_schedule(lr_cbf), weight_decay=1e-3)
        self.cbf_mlp_optim = optax.apply_if_finite(cbf_mlp_optim, 1_000_000)
        self.cbf_mlp_train_state = TrainState.create(
            apply_fn=self.cbf_mlp.apply,
            params=cbf_mlp_params,
            tx=self.cbf_mlp_optim,
        )
        self.cbf_mlp_tgt = TrainState.create(
            apply_fn=self.cbf_mlp.apply,
            params=tree_copy(cbf_mlp_params),
            tx=empty_grad_tx(),
        )

        self.key = key
        self.buffer = MaskedReplayBuffer(size=buffer_size)
        self.unsafe_buffer = MaskedTransitionBuffer(size=buffer_size // 2)
        self.rng = np.random.default_rng(seed=seed + 1)

    @property
    def config(self) -> dict:
        """Algorithm configuration dictionary."""
        return {
            'batch_size': self.batch_size,
            'lr_actor': self.lr_actor,
            'lr_cbf': self.lr_cbf,
            'alpha': self.alpha,
            'eps': self.eps,
            'inner_epoch': self.inner_epoch,
            'loss_action_coef': self.loss_action_coef,
            'loss_unsafe_coef': self.loss_unsafe_coef,
            'loss_safe_coef': self.loss_safe_coef,
            'gnn_layers': self.gnn_layers,
            'seed': self.seed,
            'max_grad_norm': self.max_grad_norm,
            'horizon': self.horizon,
            'sqp_horizon_len': self.sqp_horizon_len,
            'n_sqp_iter': self.n_sqp_iter,
            'relax_penalty': self.relax_penalty,
            'loss_dt_cbf_coef': self.loss_dt_cbf_coef,
            'loss_dyn_coef': self.loss_dyn_coef,
            'beta': self.beta,
            'cbf_tgt_tau': self.cbf_tgt_tau,
            'gamma': self.gamma,
            'labeling_horizon': self.labeling_horizon,
            'use_robust_phase2_training': self.use_robust_phase2_training,
            'robust_phase2_alpha': self.robust_phase2_alpha,
            'sqp_diagnostics': self.sqp_diagnostics,
            'sqp_filter_nonconverged': self.sqp_filter_nonconverged,
            'sqp_filter_violating': self.sqp_filter_violating,
            'sqp_dynamics': self.sqp_dynamics,
            'sqp_learned_jac': self.sqp_learned_jac,
            'sqp_learned_after': self.sqp_learned_after,
            'sqp_skip_safe': self.sqp_skip_safe,
            'sqp_exit_tol': self.sqp_exit_tol,
            'sqp_dedup': self.sqp_dedup,
            'simulator_rollout': self.simulator_rollout,
        }

    @property
    def tf_actor_params(self) -> dict:
        """Frozen parameter pytree for the CausalTransformerPolicy."""
        return self.tf_actor_train_state.params

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, unsafe_mask: Array) -> jnp.ndarray:
        """Label states as safe if no collision happens within the labeling horizon."""
        def safe_rollout(single_rollout_mask: Array) -> Array:
            safe_rollout_mask = jnp.ones_like(single_rollout_mask)
            for i in range(single_rollout_mask.shape[0]):
                start = 0 if i < self.labeling_horizon else i - self.labeling_horizon
                safe_rollout_mask = safe_rollout_mask.at[start: i + 1].set(
                    ((1 - single_rollout_mask[i]) * safe_rollout_mask[start: i + 1]).astype(jnp.bool_))
                safe_rollout_mask = safe_rollout_mask.at[0].set(1)
            return safe_rollout_mask

        safe = jax_vmap(jax_vmap(safe_rollout, in_axes=1, out_axes=1))(unsafe_mask)
        return safe

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        """Deterministic action from the GNN actor."""
        if params is None:
            params = self.actor_train_state.params
        action = 2 * self.actor.get_action(params, graph) + self._env.u_ref(graph)
        return action

    def act_transformer(
            self,
            graph: GraphsTuple,
            params: Optional[Params] = None,
    ) -> Action:
        """Single-step action from the transformer actor head."""
        if params is None:
            params = self.tf_actor_train_state.params
        obs_seq    = graph.env_states.obs_history.swapaxes(0, 1)
        action_seq = graph.env_states.action_history.swapaxes(0, 1)
        u_nom      = self._env.u_ref(graph)
        out        = self.tf_policy.apply(params, obs_seq, action_seq, u_nom, True)
        return out.u_applied

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Action:
        """Stochastic action from the GNN actor."""
        if params is None:
            params = self.actor_params
        action = self.actor_train_state.apply_fn(params, graph, key)
        return 2 * action + self._env.u_ref(graph)

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnums=1)
    def update_tgt(self, cbf_tgt: TrainState, cbf: TrainState, tau: float) -> TrainState:
        """Soft-update the target CBF network."""
        tgt_params = optax.incremental_update(cbf.params, cbf_tgt.params, tau)
        return cbf_tgt.replace(params=tgt_params)

    @ft.partial(jax.jit, static_argnums=(0,))
    def get_b_u_qp(self, b_graph: GraphsTuple, params) -> Action:
        """Batch QP action labels."""
        b_u_qp, bT_relaxation = jax_vmap(ft.partial(self.get_qp_action, cbf_params=params))(b_graph)
        return b_u_qp

    @ft.partial(jax.jit, static_argnums=(0, 2, 4, 5, 8, 10, 11))
    def get_b_u_sqp(
            self,
            b_graph: GraphsTuple,
            cbf_apply_fn: Callable,
            params,
            horizon: int = 5,
            n_sqp_iter: int = 25,
            relax_penalty: float = 1e3,
            gamma: float = 0.0,
            collect_diagnostics: bool = False,
            tf_params=None,
            dynamics_mode: str = 'true',
            jac_mode: str = 'full',
    ) -> tuple[Action, Array, Array, Array]:
        """Batch SQP action labels."""
        return jax_vmap(
            ft.partial(
                self.get_sqp_action,
                cbf_apply_fn=cbf_apply_fn,
                horizon=horizon,
                n_sqp_iter=n_sqp_iter,
                relax_penalty=relax_penalty,
                cbf_params=params,
                gamma=gamma,
                collect_diagnostics=collect_diagnostics,
                tf_params=tf_params,
                dynamics_mode=dynamics_mode,
                jac_mode=jac_mode,
            )
        )(b_graph)

    def _sqp_labels(self, b_graph: GraphsTuple, cbf_apply_fn: Callable, cbf_params, **kw):
        """SQP labels for a batch."""
        if not self.sqp_dedup:
            return self._sqp_labels_batch(b_graph, cbf_apply_fn, cbf_params, **kw)
        leaves = jtu.tree_leaves(b_graph)
        B = int(leaves[0].shape[0])
        rows = np.concatenate([np.ascontiguousarray(np.asarray(x)).reshape(B, -1).view(np.uint8)
                               for x in leaves], axis=1)
        keys = np.ascontiguousarray(rows).view(np.dtype((np.void, rows.shape[1]))).ravel()
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        n_u = first.size
        n_pad = min(B, max(256, 1 << int(np.ceil(np.log2(n_u)))))
        if n_pad >= B:
            return self._sqp_labels_batch(b_graph, cbf_apply_fn, cbf_params, **kw)
        idx_pad = np.concatenate([first, np.full(n_pad - n_u, first[0])])
        out_u = self._sqp_labels_batch(jtu.tree_map(lambda x: x[idx_pad], b_graph),
                                       cbf_apply_fn, cbf_params, **kw)
        return jtu.tree_map(lambda a: a[jnp.asarray(inverse.reshape(-1))], out_u)

    def _sqp_labels_batch(self, b_graph: GraphsTuple, cbf_apply_fn: Callable, cbf_params, **kw):
        """SQP labels for a batch, iterating only on samples that need it (sqp_skip_safe)."""
        def solve(g, n_iter):
            return self.get_b_u_sqp(g, cbf_apply_fn, cbf_params, n_sqp_iter=n_iter, **kw)

        if not self.sqp_skip_safe:
            return solve(b_graph, self.n_sqp_iter)

        out0 = solve(b_graph, 0)
        viol_before = np.asarray(out0[1])
        need = (viol_before > self.sqp_exit_tol).any(axis=(1, 2))
        idx = np.nonzero(need)[0]
        if idx.size > 0:
            B = viol_before.shape[0]
            n_pad = min(B, max(256, 1 << int(np.ceil(np.log2(idx.size)))))
            idx_pad = np.concatenate([idx, np.full(n_pad - idx.size, idx[0])])
            sub = solve(jtu.tree_map(lambda x: x[idx_pad], b_graph), self.n_sqp_iter)
            out0 = jtu.tree_map(lambda a, s: a.at[idx].set(s[:idx.size]), out0, sub)
        if self.sqp_diagnostics:
            out0 = (*out0[:4], out0[4]._replace(budget_hit=out0[3] >= self.n_sqp_iter))
        return out0

    def update_nets(self, rollout: Rollout, safe_mask, unsafe_mask, step: int = 0):
        """Compute SQP labels and run the inner update epochs."""
        update_info = {}

        cbf_apply_fn = self.cbf_mlp.apply

        use_learned_dyn = (
            self.sqp_dynamics == 'learned' and step >= self.sqp_learned_after
        )
        dynamics_mode = 'learned' if use_learned_dyn else 'true'
        if use_learned_dyn and not self._sqp_learned_announced:
            print(f"  [SQP teacher] step {step}: transition model -> LEARNED "
                  f"(dynamics head, jac={self.sqp_learned_jac}).  "
                  f"u_nom and observation reconstruction remain analytic.")
            self._sqp_learned_announced = True

        sqp_out = self._sqp_labels(
            rollout.graph,
            cbf_apply_fn,
            self.cbf_mlp_tgt.params,
            horizon=self.sqp_horizon_len,
            relax_penalty=self.relax_penalty,
            gamma=self.gamma,
            collect_diagnostics=self.sqp_diagnostics,
            tf_params=self.tf_actor_train_state.params if use_learned_dyn else None,
            dynamics_mode=dynamics_mode,
            jac_mode=self.sqp_learned_jac,
        )
        if self.sqp_diagnostics:
            (b_u_sqp, b_violation_before, b_violation_after,
             b_n_iter_used, b_diag) = sqp_out
        else:
            b_u_sqp, b_violation_before, b_violation_after, b_n_iter_used = sqp_out
            b_diag = None

        sqp_nan = bool(jnp.isnan(b_u_sqp).any())
        sqp_inf = bool(jnp.isinf(b_u_sqp).any())
        if sqp_nan or sqp_inf:
            import warnings
            warnings.warn(
                f"[update_nets] SQP labels contain {'NaN' if sqp_nan else 'Inf'} — "
                "apply_if_finite will skip ALL gradient updates this step. "
                "Check SQP convergence (n_sqp_iter, relax_penalty) or action bounds."
            )

        n_processed      = len(rollout.graph.states)
        safe_mask_trim   = safe_mask[:n_processed]
        unsafe_mask_trim = unsafe_mask[:n_processed]

        sqp_conv_thresh = SQP_CONV_THRESH
        sqp_converged = (b_violation_after.max(axis=(-2, -1)) < sqp_conv_thresh)

        lact_mask = sqp_converged
        excl_nonconverged = None
        excl_violating    = None
        if b_diag is not None:
            excl_nonconverged = np.asarray(jnp.logical_not(b_diag.converged_strict))
            excl_violating    = np.asarray(b_diag.violates_untightened)
            if self.sqp_filter_nonconverged:
                lact_mask = jnp.logical_and(lact_mask, b_diag.converged_strict)
            if self.sqp_filter_violating:
                lact_mask = jnp.logical_and(
                    lact_mask, jnp.logical_not(b_diag.violates_untightened))

        batch_orig = Batch(rollout.graph, safe_mask_trim, unsafe_mask_trim, b_u_sqp, lact_mask)

        for i_epoch in range(self.inner_epoch):
            n_batches = n_processed // self.batch_size
            if n_batches == 0:
                break
            idx       = self.rng.choice(n_processed, size=n_batches * self.batch_size, replace=False)
            batch_idx = idx.reshape(n_batches, self.batch_size)
            batch     = jtu.tree_map(lambda x: x[batch_idx], batch_orig)

            cbf_mlp_ts, tf_actor_ts, update_info = self.update_inner_transformer(
                self.cbf_mlp_train_state, self.tf_actor_train_state, batch
            )
            self.cbf_mlp_train_state  = cbf_mlp_ts
            self.tf_actor_train_state = tf_actor_ts

        self.cbf_mlp_tgt = self.update_tgt(self.cbf_mlp_tgt, self.cbf_mlp_train_state, self.cbf_tgt_tau)

        update_info['sqp/violation_before_max']  = float(b_violation_before.max())
        update_info['sqp/violation_before_mean'] = float(b_violation_before.mean())
        update_info['sqp/violation_after_max']   = float(b_violation_after.max())
        update_info['sqp/violation_after_mean']  = float(b_violation_after.mean())
        update_info['sqp/n_iter_used_mean']      = float(b_n_iter_used.mean())
        update_info['sqp/converged_frac']        = float(
            jnp.mean(b_violation_after.max(axis=(-2, -1)) < sqp_conv_thresh)
        )

        n_samples   = int(b_violation_after.shape[0])
        solved_cnt  = int(sqp_converged.sum())
        resid_after = b_violation_after.max(axis=(-2, -1))
        failed_mask = jnp.logical_not(sqp_converged)
        n_failed    = int(failed_mask.sum())
        budget_hit  = (b_n_iter_used >= self.n_sqp_iter)

        update_info['sqp/n_samples']       = n_samples
        update_info['sqp/solved_count']    = solved_cnt
        update_info['sqp/failed_count']    = n_samples - solved_cnt
        update_info['sqp/failed_frac']     = 1.0 - update_info['sqp/converged_frac']
        update_info['sqp/budget_hit_frac'] = float(jnp.mean(budget_hit))
        update_info['sqp/n_iter_used_max'] = float(b_n_iter_used.max())
        update_info['sqp/violation_after_mean_failed'] = float(
            jnp.where(failed_mask, resid_after, 0.0).sum() / max(n_failed, 1)
        )

        if b_diag is not None:
            row = sqp_diag.aggregate(
                jtu.tree_map(np.asarray, b_diag),
                env=self.sqp_diag_env_name,
                horizon=self.sqp_horizon_len,
                seed=self.seed,
                train_step=int(step),
                eta_s=self.gamma,
                lambda_xi=self.relax_penalty,
                alpha_decay=self.alpha,
                n_sqp_iter=self.n_sqp_iter,
                excluded_mask=np.asarray(jnp.logical_not(lact_mask)),
                excluded_nonconverged=excl_nonconverged,
                excluded_violating=excl_violating,
            )
            self._sqp_diag_rows.append(row)
            update_info['sqp/converged_strict_frac']   = row['converged_strict_frac']
            update_info['sqp/xi_nonzero_frac']         = row['xi_nonzero_frac']
            update_info['sqp/xi_p95']                  = row['xi_p95']
            update_info['sqp/violates_untightened_frac'] = row['violates_untightened_frac']
            update_info['sqp/xi_below_eta_s_frac']     = row['xi_below_eta_s_frac']
            update_info['sqp/excluded_from_lact_frac'] = row['excluded_from_lact_frac']
            update_info['sqp/nonfinite_frac']          = row['nonfinite_frac']
            update_info['sqp/qp_degraded_frac']        = row['qp_degraded_frac']

        return update_info

    @property
    def sqp_diag_rows(self) -> list:
        """Accumulated per-update diagnostic rows (one per update_nets call)."""
        return self._sqp_diag_rows

    def write_sqp_diagnostics(self, path: Optional[str] = None) -> Optional[str]:
        """Write accumulated rows to CSV."""
        if not self._sqp_diag_rows:
            return None
        path = path or self.sqp_diag_csv
        sqp_diag.write_csv(self._sqp_diag_rows, path)
        return path

    def print_sqp_diagnostics_summary(self) -> None:
        """Print the end-of-training per-environment summary block."""
        if not self._sqp_diag_rows:
            return
        print()
        print(sqp_diag.format_summary(self._sqp_diag_rows[-1]))

    @staticmethod
    def _ts_to_dict(ts) -> dict:
        """Picklable subset of a TrainState (no tx closures)."""
        return {'params': ts.params, 'opt_state': ts.opt_state, 'step': ts.step}

    def save(self, save_dir: str, step: int):
        """Save the actor, CBF and target CBF parameters."""
        model_dir = os.path.join(save_dir, str(step))
        os.makedirs(model_dir, exist_ok=True)

        pickle.dump(
            self._ts_to_dict(self.tf_actor_train_state),
            open(os.path.join(model_dir, 'actor.pkl'), 'wb'),
        )
        pickle.dump(
            self._ts_to_dict(self.cbf_mlp_train_state),
            open(os.path.join(model_dir, 'cbf.pkl'), 'wb'),
        )
        pickle.dump(
            self.cbf_mlp_tgt.params,
            open(os.path.join(model_dir, 'cbf_tgt.pkl'), 'wb'),
        )

    def load(self, load_dir: str, step: int):
        """Load the actor, CBF and target CBF parameters."""
        path = os.path.join(load_dir, str(step))

        d = pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb'))
        self.tf_actor_train_state = self.tf_actor_train_state.replace(
            params=d['params'], opt_state=d['opt_state'], step=d['step']
        )

        d = pickle.load(open(os.path.join(path, 'cbf.pkl'), 'rb'))
        self.cbf_mlp_train_state = self.cbf_mlp_train_state.replace(
            params=d['params'], opt_state=d['opt_state'], step=d['step']
        )

        tgt_path = os.path.join(path, 'cbf_tgt.pkl')
        tgt_params = (
            pickle.load(open(tgt_path, 'rb'))
            if os.path.exists(tgt_path)
            else self.cbf_mlp_train_state.params
        )
        self.cbf_mlp_tgt = self.cbf_mlp_tgt.replace(params=tgt_params)

    def sample_batch(self, rollout: Rollout, safe_mask, unsafe_mask):
        """Update batch = [replay memory | unsafe memory | current rollout], as in GCBF+."""
        n_cur = rollout.length * rollout.time_horizon

        flat_rollout = jtu.tree_map(lambda x: merge01(x), rollout)
        flat_safe    = merge01(safe_mask)
        flat_unsafe  = merge01(unsafe_mask)

        parts = []
        if self.buffer.length > 0:
            parts.append(self.buffer.sample(n_cur))
        if self.unsafe_buffer.length > 0:
            parts.append(self.unsafe_buffer.sample(n_cur))
        parts.append((flat_rollout, flat_safe, flat_unsafe))

        self.buffer.append(rollout, safe_mask, unsafe_mask)
        step_unsafe = np.asarray(unsafe_mask).max(axis=-1).astype(bool)
        if step_unsafe.any():
            self.unsafe_buffer.append(
                jtu.tree_map(lambda x: x[step_unsafe], rollout),
                safe_mask[step_unsafe],
                unsafe_mask[step_unsafe],
            )

        rollout_batch     = tree_merge([p[0] for p in parts])
        safe_mask_batch   = tree_merge([p[1] for p in parts])
        unsafe_mask_batch = tree_merge([p[2] for p in parts])
        return rollout_batch, safe_mask_batch, unsafe_mask_batch

    def update(self, rollout: Rollout, step: int) -> dict:
        """One training update from a rollout."""
        key, self.key = jr.split(self.key)

        unsafe_mask = jax_vmap(jax_vmap(self._env.unsafe_mask))(rollout.graph)
        safe_mask = self.safe_mask(unsafe_mask)
        safe_mask, unsafe_mask = jax2np(safe_mask), jax2np(unsafe_mask)

        rollout_np = jax2np(rollout)
        del rollout
        rollout_batch, safe_mask_batch, unsafe_mask_batch = self.sample_batch(rollout_np, safe_mask, unsafe_mask)

        update_info = self.update_nets(rollout_batch, safe_mask_batch, unsafe_mask_batch, step=step)

        return update_info

    def get_sqp_action(
            self,
            graph: GraphsTuple,
            cbf_apply_fn: Callable,
            horizon: int = 5,
            n_sqp_iter: int = 25,
            relax_penalty: float = 1e3,
            cbf_params=None,
            alpha: Optional[float] = None,
            gamma: float = 0.0,
            qp_settings: JaxProxQP.Settings = None,
            collect_diagnostics: bool = False,
            tf_params=None,
            dynamics_mode: str = 'true',
            jac_mode: str = 'full',
    ) -> tuple[Action, Array, Array, Array]:

        """SQP teacher: optimize an H-step action sequence subject to the CBF constraints."""
        if alpha is None:
            alpha = self.alpha
        assert graph.is_single

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.n_agents)
        obstacles   = graph.env_states.obstacle

        u_nom_t    = self._env.u_ref(graph)

        u_lb_scalar, u_ub_scalar = self._env.action_lim()
        u_lb_tiled = jnp.tile(u_lb_scalar, horizon * self.n_agents)
        u_ub_tiled = jnp.tile(u_ub_scalar, horizon * self.n_agents)

        n_u = horizon * self.n_agents * self._env.action_dim
        n_r = horizon * self.n_agents
        n_c = n_r

        delta_U = jnp.zeros((horizon, self.n_agents, self._env.action_dim), dtype=jnp.float32)
        r_init  = jnp.zeros(n_r, dtype=jnp.float32)

        if qp_settings is None:
            qp_settings = JaxProxQP.Settings.default()
        qp_settings.dua_gap_thresh_abs = None

        obs_from_state: Callable = make_obs_from_state_fn(
            goal_states=goal_states,
            obstacles=obstacles,
            n_rays=self._env._params["n_rays"],
            comm_radius=self._env._params["comm_radius"],
            num_agents=self.n_agents,
            ray_order=self._env._params.get("lidar_ray_order", False),
            state_obs=self._env._params.get("state_obs", "raw"),
            env=self._env,
        )

        def h_from_state(x_k: Array) -> Array:
            o_k = obs_from_state(x_k)
            h_k = cbf_apply_fn(cbf_params, o_k)
            return h_k.reshape(self.n_agents)

        def c_of_delta_U(delta_U_in: Array) -> tuple[Array, Array]:

            def scan_fn(x_carry, delta_u_k):
                u_nom_k   = self._env.u_nom_from_state(x_carry, goal_states)
                u_applied = u_nom_k + delta_u_k
                x_next_raw = self._env.step_simulator(x_carry, u_applied)
                if hasattr(self._env, "true_next_state"):
                    x_fwd = self._env.true_next_state(x_carry, u_applied, goal_states)
                else:
                    x_fwd = self._env.clip_state(x_next_raw)
                x_next     = x_next_raw + jax.lax.stop_gradient(x_fwd - x_next_raw)
                return x_next, x_next

            _, x_hat_steps = jax.lax.scan(
                scan_fn,
                agent_state,
                delta_U_in,
            )

            x_hat_all = jnp.concatenate(
                [agent_state[None], x_hat_steps], axis=0
            )

            h_vals = jax.vmap(h_from_state)(x_hat_all)

            c = gamma - h_vals[1:] + (1.0 - alpha) * h_vals[:-1]

            return c, x_hat_all

        def c_of_delta_U_learned(delta_U_in: Array) -> tuple[Array, Array]:

            def scan_fn(carry, delta_u_k):
                x_carry, obs_k, act_k = carry

                u_nom_k   = self._env.u_nom_from_state(x_carry, goal_states)
                u_applied = u_nom_k + delta_u_k

                u_clip_raw = jnp.clip(u_applied, u_lb_scalar[None], u_ub_scalar[None])
                u_clip_k   = u_applied + jax.lax.stop_gradient(u_clip_raw - u_applied)

                out = self.tf_policy.apply(
                    tf_params_sg, obs_k, act_k, u_nom_k, True,
                )
                delta_x_hat = self.tf_policy.apply(
                    tf_params_sg, out.z_t, u_clip_k,
                    method=lambda mdl, zz, uu: mdl.call_dynamics_head(zz, uu),
                )

                x_next_raw = x_carry + delta_x_hat
                x_next     = x_next_raw + jax.lax.stop_gradient(
                    self._env.clip_state(x_next_raw) - x_next_raw
                )

                o_live = obs_from_state(x_next)
                if jac_mode == 'state-only':
                    o_ctx = jax.lax.stop_gradient(o_live)
                    u_ctx = jax.lax.stop_gradient(u_clip_k)
                else:
                    o_ctx, u_ctx = o_live, u_clip_k

                obs_next = jnp.concatenate([obs_k[:, 1:, :], o_ctx[:, None, :]], axis=1)
                act_next = jnp.concatenate([act_k[:, 1:, :], u_ctx[:, None, :]], axis=1)

                return (x_next, obs_next, act_next), (x_next, o_live)

            _, (x_hat_steps, o_hat_steps) = jax.lax.scan(
                scan_fn,
                (agent_state, obs_seq_0, act_seq_0),
                delta_U_in,
            )

            x_hat_all = jnp.concatenate(
                [agent_state[None], x_hat_steps], axis=0
            )

            obs_all = jnp.concatenate(
                [obs_from_state(agent_state)[None], o_hat_steps], axis=0
            )

            h_vals = jax.vmap(
                lambda o_k: cbf_apply_fn(cbf_params, o_k).reshape(self.n_agents)
            )(obs_all)

            c = gamma - h_vals[1:] + (1.0 - alpha) * h_vals[:-1]

            return c, x_hat_all

        if dynamics_mode == 'learned':
            if tf_params is None:
                raise ValueError(
                    "dynamics_mode='learned' requires tf_params (the "
                    "CausalTransformerPolicy parameter pytree); got None."
                )
            tf_params_sg = jtu.tree_map(jax.lax.stop_gradient, tf_params)

            obs_seq_0 = graph.env_states.obs_history.swapaxes(0, 1)
            act_seq_0 = graph.env_states.action_history.swapaxes(0, 1)

            c_of_delta_U = c_of_delta_U_learned

        c_init, x_hat_init_actual = c_of_delta_U(delta_U)
        violation_before = jnp.maximum(c_init, 0.0)

        def body_fn(carry: tuple) -> tuple:
            delta_U_i, r_carry, x_hat_all, i, c_nom = carry[:5]

            c_nom_flat = c_nom.reshape(-1)

            A_cbf_raw = jax.jacobian(
                lambda dU: c_of_delta_U(dU)[0]
            )(delta_U_i)
            A_cbf = A_cbf_raw.reshape(n_c, n_u)

            H_mat = jnp.eye(n_u + n_r, dtype=jnp.float32)
            H_mat = H_mat.at[-n_r:, -n_r:].set(
                H_mat[-n_r:, -n_r:] * relax_penalty
            )

            g_vec = jnp.concatenate([
                delta_U_i.reshape(-1).astype(jnp.float32),
                jnp.zeros(n_r, dtype=jnp.float32),
            ])

            C_mat = jnp.concatenate(
                [A_cbf, -jnp.eye(n_c, dtype=jnp.float32)], axis=1
            )
            b_vec = -c_nom_flat.astype(jnp.float32)

            u_nom_traj = jax.vmap(
                lambda x_k: self._env.u_nom_from_state(x_k, goal_states)
            )(x_hat_all[:-1])
            u_nom_traj_flat = u_nom_traj.reshape(-1).astype(jnp.float32)
            delta_lb = (u_lb_tiled - u_nom_traj_flat - delta_U_i.reshape(-1)).astype(jnp.float32)
            delta_ub = (u_ub_tiled - u_nom_traj_flat - delta_U_i.reshape(-1)).astype(jnp.float32)
            l_box = jnp.concatenate([delta_lb, jnp.zeros(n_r, dtype=jnp.float32)])
            u_box = jnp.concatenate([delta_ub, jnp.full(n_r, jnp.inf, dtype=jnp.float32)])

            qp     = JaxProxQP.QPModel.create(H_mat, g_vec, C_mat, b_vec, l_box, u_box)
            solver = JaxProxQP(qp, qp_settings)
            sol    = solver.solve()

            delta_flat  = sol.x[:n_u]
            delta_U_new = delta_U_i + delta_flat.reshape(horizon, self.n_agents, self._env.action_dim)

            c_new, x_hat_new = c_of_delta_U(delta_U_new)

            if collect_diagnostics:
                xi_qp = sol.x[n_u:]
                return (delta_U_new, xi_qp, x_hat_new, i + 1, c_new,
                        sol.pri_res, sol.dua_res)

            return (delta_U_new, r_init, x_hat_new, i + 1, c_new)

        def cond_fn(carry: tuple) -> bool:
            i, c = carry[3], carry[4]
            return jnp.logical_and(i < n_sqp_iter, jnp.any(c > self.sqp_exit_tol))

        init_carry = (delta_U, r_init, x_hat_init_actual,
                      jnp.array(0, dtype=jnp.int32), c_init)
        if collect_diagnostics:
            init_carry = init_carry + (jnp.array(0.0, dtype=jnp.float32),
                                       jnp.array(0.0, dtype=jnp.float32))
        final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        delta_U_final, xi_qp_final, _, n_iter_used, c_final = final_carry[:5]

        violation_after = jnp.maximum(c_final, 0.0)

        u_opt = u_nom_t + delta_U_final[0]
        u_opt = jnp.clip(u_opt, u_lb_scalar[None], u_ub_scalar[None])

        if not collect_diagnostics:
            return u_opt, violation_before, violation_after, n_iter_used

        eta_s = gamma
        qp_pri_res, qp_dua_res = final_carry[5], final_carry[6]

        converged_strict = jnp.all(c_final <= 0.0)
        converged_thresh = violation_after.max() < SQP_CONV_THRESH
        budget_hit       = n_iter_used >= n_sqp_iter

        has_nonfinite = jnp.logical_not(
            jnp.logical_and(jnp.all(jnp.isfinite(c_final)),
                            jnp.all(jnp.isfinite(delta_U_final)))
        )

        xi_real          = violation_after
        xi_max           = xi_real.max()
        xi_sum           = xi_real.sum()
        xi_nonzero_count = jnp.sum(xi_real > SQP_SLACK_TOL).astype(jnp.int32)

        untightened_margin_max = (c_final - eta_s).max()
        violates_untightened   = untightened_margin_max > 0.0
        xi_below_eta_s = jnp.logical_and(xi_max > SQP_SLACK_TOL, xi_max < eta_s)

        safe_eta_s = jnp.where(eta_s > 0.0, eta_s, jnp.nan)
        xi_max_over_eta_s = jnp.where(eta_s > 0.0, xi_max / safe_eta_s, jnp.inf)

        xi_qp                = jnp.abs(xi_qp_final)
        xi_qp_max            = xi_qp.max()
        xi_qp_sum            = xi_qp.sum()
        xi_qp_nonzero_count  = jnp.sum(xi_qp > SQP_SLACK_TOL).astype(jnp.int32)
        xi_qp_max_over_eta_s = jnp.where(
            eta_s > 0.0, xi_qp_max / safe_eta_s, jnp.inf)
        xi_qp_exceeds_eta_s  = xi_qp_max > eta_s

        qp_degraded = jnp.logical_or(qp_pri_res > QP_PRI_RES_THRESH,
                                     qp_dua_res > QP_DUA_RES_THRESH)

        exit_code = jnp.where(
            has_nonfinite, EXIT_NONFINITE,
            jnp.where(converged_strict, EXIT_CONVERGED, EXIT_MAX_ITERS),
        ).astype(jnp.int32)

        diagnostics = SQPSolveDiagnostics(
            converged_strict=converged_strict,
            converged_thresh=converged_thresh,
            n_iters=n_iter_used.astype(jnp.int32),
            budget_hit=budget_hit,
            exit_code=exit_code,
            has_nonfinite=has_nonfinite,
            xi_max=xi_max,
            xi_sum=xi_sum,
            xi_nonzero_count=xi_nonzero_count,
            xi_max_over_eta_s=xi_max_over_eta_s,
            xi_below_eta_s=xi_below_eta_s,
            violates_untightened=violates_untightened,
            untightened_margin_max=untightened_margin_max,
            xi_qp_max=xi_qp_max,
            xi_qp_sum=xi_qp_sum,
            xi_qp_nonzero_count=xi_qp_nonzero_count,
            xi_qp_max_over_eta_s=xi_qp_max_over_eta_s,
            xi_qp_exceeds_eta_s=xi_qp_exceeds_eta_s,
            qp_pri_res=qp_pri_res,
            qp_dua_res=qp_dua_res,
            qp_degraded=qp_degraded,
        )
        return u_opt, violation_before, violation_after, n_iter_used, diagnostics

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 2))
    def update_inner_transformer(
            self,
            cbf_mlp_train_state: TrainState,
            tf_actor_train_state: TrainState,
            batch: Batch,
    ) -> tuple[TrainState, TrainState, dict]:
        
        """Minibatch gradient updates of the transformer policy and CBF MLP."""
        def update_fn(carry, minibatch: Batch):
            cbf_mlp, tf_actor, dropout_key = carry
            dropout_key, step_key = jr.split(dropout_key)

            safe_mask_flat   = merge01(minibatch.safe_mask)
            unsafe_mask_flat = merge01(minibatch.unsafe_mask)
            obs_current = minibatch.graph.env_states.obs_history[:, -1, :, :]
            obs_seq    = minibatch.graph.env_states.obs_history.transpose(0, 2, 1, 3)
            action_seq = minibatch.graph.env_states.action_history.transpose(0, 2, 1, 3)

            u_nom_batch = jax.vmap(self._env.u_ref)(minibatch.graph)

            goal_batch     = minibatch.graph.env_states.goal
            obstacle_batch = minibatch.graph.env_states.obstacle

            def get_loss(cbf_mlp_params, tf_actor_params):
                labeled_mask      = jnp.logical_or(minibatch.safe_mask, minibatch.unsafe_mask)
                labeled_mask_flat = merge01(labeled_mask)

                tf_params_dyn_sg = {
                    'params': {
                        **{k: v for k, v in tf_actor_params['params'].items() if k != 'dynamics_head'},
                        'dynamics_head': jtu.tree_map(
                            jax.lax.stop_gradient,
                            tf_actor_params['params']['dynamics_head'],
                        ),
                    }
                }

                def eval_cbf(o):
                    return self.cbf_mlp.apply(cbf_mlp_params, o).reshape(self.n_agents)

                def eval_cbf_stopped(o):
                    return self.cbf_mlp.apply(jax.lax.stop_gradient(cbf_mlp_params), o).reshape(self.n_agents)

                h      = jax.vmap(eval_cbf)(obs_current)
                h_flat = merge01(h)

                unsafe_count = jnp.count_nonzero(unsafe_mask_flat) + 1e-6
                safe_count   = jnp.count_nonzero(safe_mask_flat)   + 1e-6

                h_unsafe = jnp.where(unsafe_mask_flat, h_flat,-self.eps * 2 * jnp.ones_like(h_flat),)
                loss_unsafe = jnp.sum(jax.nn.relu(h_unsafe + self.eps)) / unsafe_count
                acc_unsafe_mask = jnp.where(unsafe_mask_flat, h_flat, jnp.ones_like(h_flat))
                acc_unsafe = (jnp.sum(jnp.less(acc_unsafe_mask, 0)) + 1e-6) / unsafe_count

                h_safe = jnp.where(safe_mask_flat, h_flat,self.eps * 2 * jnp.ones_like(h_flat),)
                loss_safe = jnp.sum(jax.nn.relu(-h_safe + self.eps)) / safe_count
                acc_safe_mask = jnp.where(safe_mask_flat, h_flat, -jnp.ones_like(h_flat))
                acc_safe = (jnp.sum(jnp.greater(acc_safe_mask, 0)) + 1e-6) / safe_count

                def tf_fwd(obs_s, act_s, u_n):
                    out = self.tf_policy.apply(
                        tf_actor_params, obs_s, act_s, u_n, True,
                        rngs={'dropout': step_key},
                    )
                    return out.delta_u, out.u_applied, out.z_t

                delta_u_batch, u_applied_batch, z_t_batch = jax.vmap(tf_fwd)(
                    obs_seq, action_seq, u_nom_batch
                )

                u_sqp         = minibatch.u_qp
                delta_a_label = u_sqp - u_nom_batch

                per_sample_loss = jnp.sum(jnp.square(delta_u_batch - delta_a_label), axis=-1)
                per_sample_loss = per_sample_loss.mean(axis=-1)
                if minibatch.sqp_converged is not None:
                    conv_mask = minibatch.sqp_converged.astype(jnp.float32)
                    n_converged = jnp.maximum(conv_mask.sum(), 1.0)
                    loss_action = (per_sample_loss * conv_mask).sum() / n_converged
                else:
                    loss_action = per_sample_loss.mean()

                x_t = minibatch.graph.env_states.agent
                u_lb_s, u_ub_s = self._env.action_lim()
                u_clipped = jnp.clip(u_applied_batch, u_lb_s[None, None, :], u_ub_s[None, None, :])

                if hasattr(self._env, "true_next_state"):
                    _true_next = self._env.true_next_state
                else:
                    _true_next = lambda xi, ui, gi: self._env.clip_state(self._env.step_simulator(xi, ui))
                x_next_true  = jax.lax.stop_gradient(
                    jax.vmap(_true_next)(x_t, u_clipped, goal_batch)
                )
                delta_x_true = x_next_true - x_t

                u_applied_sg = jax.lax.stop_gradient(u_clipped)

                def dyn_only(z, u):
                    return self.tf_policy.apply(
                        tf_actor_params, z, u,
                        method=lambda mdl, zz, uu: mdl.call_dynamics_head(zz, uu),
                    )

                delta_x_hat_for_dyn = jax.vmap(dyn_only)(z_t_batch, u_applied_sg)
                loss_dyn     = jnp.mean(jnp.sum(jnp.square(delta_x_hat_for_dyn - delta_x_true), axis=-1))

                delta_x_hat_single = jax.vmap(
                    lambda z, u: self.tf_policy.apply(
                        tf_params_dyn_sg, z, u,
                        method=lambda mdl, zz, uu: mdl.call_dynamics_head(zz, uu),
                    )
                )(z_t_batch, u_clipped)
                x_hat_1 = self._env.clip_state(x_t + delta_x_hat_single)

                def _sim_step(x_b, u_b):
                    def one(xi, ui, gi):
                        raw = self._env.step_simulator(xi, ui)
                        if hasattr(self._env, "true_next_state"):
                            fwd = self._env.true_next_state(xi, ui, gi)
                        else:
                            fwd = self._env.clip_state(raw)
                        return raw + jax.lax.stop_gradient(fwd - raw)
                    return jax.vmap(one)(x_b, u_b, goal_batch)

                if self.simulator_rollout:
                    x_hat_1 = self._env.clip_state(_sim_step(x_t, u_clipped))

                def obs_next_fn(x_next_i, goal_i, obstacle_i):
                    return self._env._get_obs(x_next_i, goal_i, obstacle_i)

                obs_next = jax.vmap(obs_next_fn)(x_hat_1, goal_batch, obstacle_batch)

                h_next_full    = jax.vmap(eval_cbf)(obs_next)
                h_next_stopped = jax.vmap(eval_cbf_stopped)(obs_next)
                h_next = jnp.where(labeled_mask, h_next_full, h_next_stopped)

                c_single    = self.gamma - h_next + (1.0 - self.alpha) * h
                loss_dt_cbf = jnp.mean(jax.nn.relu(c_single + self.eps))
                acc_dt_cbf  = jnp.mean(c_single <= 0.0)

                u_lb_s, u_ub_s = self._env.action_lim()
                tf_horizon = self.horizon

                def _tf_fwd_roll(obs_s, act_s, u_n):
                    out = self.tf_policy.apply(
                        tf_params_dyn_sg, obs_s, act_s, u_n, True,
                        rngs={'dropout': step_key},
                    )
                    return out.u_applied, out.delta_x_hat

                robust_noise_key0 = jr.fold_in(step_key, 0x5D0B)

                def scan_step(carry, _):
                    if self.use_robust_phase2_training:
                        x_k, obs_k, act_k, noise_key = carry
                    else:
                        x_k, obs_k, act_k = carry

                    u_nom_k = jax.vmap(self._env.u_nom_from_state)(x_k, goal_batch)
                    u_app_k, dx_hat_k = jax.vmap(_tf_fwd_roll)(obs_k, act_k, u_nom_k)

                    u_clip_k = jnp.clip(u_app_k, u_lb_s[None, None, :], u_ub_s[None, None, :])

                    x_next = x_k + dx_hat_k
                    if self.simulator_rollout:
                        x_next = _sim_step(x_k, u_clip_k)

                    if self.use_robust_phase2_training:
                        noise_key, sub = jr.split(noise_key)
                        disturbance = self.robust_phase2_alpha * jr.normal(
                            sub, x_next.shape, dtype=x_next.dtype
                        )
                        x_next = x_next + disturbance

                    x_k = self._env.clip_state(x_next)

                    o_hat_k = jax.vmap(obs_next_fn)(x_k, goal_batch, obstacle_batch)
                    obs_k = jnp.concatenate([obs_k[:, :, 1:, :], o_hat_k[:, :, None, :]], axis=2)
                    act_k = jnp.concatenate([act_k[:, :, 1:, :], u_clip_k[:, :, None, :]], axis=2)
                    if self.use_robust_phase2_training:
                        return (x_k, obs_k, act_k, noise_key), x_k
                    return (x_k, obs_k, act_k), x_k

                if self.use_robust_phase2_training:
                    scan_init = (x_t, obs_seq, action_seq, robust_noise_key0)
                else:
                    scan_init = (x_t, obs_seq, action_seq)

                _, x_hat_steps = lax.scan(
                    scan_step,
                    scan_init,
                    None,
                    length=tf_horizon,
                )

                x_hat_horizon_tf = jnp.concatenate(
                    [x_t[:, None, :, :], x_hat_steps.transpose(1, 0, 2, 3)],
                    axis=1,
                )

                def eval_horizon_sample(x_h_i, goal_i, obstacle_i, labeled_i):

                    def h_at(x_k):

                        o_k = self._env._get_obs(x_k, goal_i, obstacle_i)
                        return self.cbf_mlp.apply(cbf_mlp_params, o_k).reshape(self.n_agents)

                    def h_at_stop_cbf(x_k):

                        o_k = self._env._get_obs(x_k, goal_i, obstacle_i)
                        return self.cbf_mlp.apply(
                            jax.lax.stop_gradient(cbf_mlp_params), o_k
                        ).reshape(self.n_agents)

                    h_seq_full = jax.vmap(h_at)(x_h_i)
                    h_seq_stop = jax.vmap(h_at_stop_cbf)(x_h_i)

                    h_rollout = jnp.where(
                        labeled_i[None, :],
                        h_seq_full[1:],
                        h_seq_stop[1:],
                    )
                    h_current = jnp.where(
                        labeled_i[None, :],
                        h_seq_full[:-1],
                        h_seq_stop[:-1],
                    )

                    c_k = self.gamma - h_rollout + (1.0 - self.alpha) * h_current
                    v_k = jax.nn.relu(c_k + self.eps)
                    H_len = v_k.shape[0]
                    return (
                        jax.scipy.special.logsumexp(self.beta * v_k, axis=0)
                        / self.beta
                        - jnp.log(H_len) / self.beta
                    )

                horizon_viol = jax.vmap(eval_horizon_sample)(
                    x_hat_horizon_tf,
                    goal_batch,
                    obstacle_batch,
                    labeled_mask,
                )
                loss_dt_cbf_horizon = jnp.mean(horizon_viol)

                total_loss = (
                    self.loss_action_coef  * loss_action
                    + self.loss_dyn_coef   * loss_dyn
                    + self.loss_unsafe_coef * loss_unsafe
                    + self.loss_safe_coef   * loss_safe
                    + self.loss_dt_cbf_coef * (loss_dt_cbf + loss_dt_cbf_horizon)
                )

                return total_loss, {
                    'loss_tf/action':          loss_action,
                    'loss_tf/dyn':             loss_dyn,
                    'loss_tf/unsafe':          loss_unsafe,
                    'loss_tf/safe':            loss_safe,
                    'loss_tf/dt_cbf':          loss_dt_cbf,
                    'loss_tf/dt_cbf_horizon':  loss_dt_cbf_horizon,
                    'loss_tf/total':           total_loss,
                    'acc_tf/unsafe':           acc_unsafe,
                    'acc_tf/safe':             acc_safe,
                    'acc_tf/dt_cbf':           acc_dt_cbf,
                }

            (loss, loss_info), (grad_cbf_mlp, grad_tf) = jax.value_and_grad(
                get_loss, has_aux=True, argnums=(0, 1)
            )(cbf_mlp.params, tf_actor.params)

            grad_cbf_mlp, g_cbf_norm = compute_norm_and_clip(grad_cbf_mlp, self.max_grad_norm)
            grad_tf,      g_tf_norm  = compute_norm_and_clip(grad_tf,      self.max_grad_norm)

            cbf_mlp  = cbf_mlp.apply_gradients(grads=grad_cbf_mlp)
            tf_actor = tf_actor.apply_gradients(grads=grad_tf)

            grad_info = {
                'grad_norm_tf/cbf_mlp':  g_cbf_norm,
                'grad_norm_tf/tf_actor': g_tf_norm,
            }
            return (cbf_mlp, tf_actor, dropout_key), grad_info | loss_info

        init_dropout_key, _ = jr.split(self.key)
        (cbf_mlp_train_state, tf_actor_train_state, _), info = lax.scan(
            update_fn,
            (cbf_mlp_train_state, tf_actor_train_state, init_dropout_key),
            batch,
        )

        info_mean = jtu.tree_map(lambda x: jnp.mean(x, axis=0), info)
        info_last = jtu.tree_map(lambda x: x[-1], info)
        info_combined = {
            **{f"mean/{k}": v for k, v in info_mean.items()},
            **info_last,
        }
        return cbf_mlp_train_state, tf_actor_train_state, info_combined

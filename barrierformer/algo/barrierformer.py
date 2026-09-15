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
from barrierformer.trainer.buffer import MaskedReplayBuffer
from barrierformer.trainer.utils import compute_norm_and_clip, jax2np, tree_copy, empty_grad_tx
from barrierformer.env.base import MultiAgentEnv
from barrierformer.algo.module.cbf import CBF
from barrierformer.algo.module.policy import DeterministicPolicy
from barrierformer.algo.rollout import make_obs_from_state_fn
from barrierformer.nn.model import CausalTransformerPolicy, TransformerPolicyConfig
from barrierformer.nn.cbf import CBFMLP
from .gcbf import GCBF


class Batch(NamedTuple):
    graph: GraphsTuple
    safe_mask: Array
    unsafe_mask: Array
    u_qp: Action
    # True where SQP converged (violation_after < threshold) — action imitation
    # is suppressed for samples where the teacher itself couldn't find a safe action.
    # Shape [B] or None (None = treat all samples as converged, backward compat).
    sqp_converged: Optional[Array] = None


class BarrierFormer(GCBF):

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
        super(GCBF, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        # set hyperparameters
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

        # set nominal graph for initialization of the neural networks
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
        # ── end GNN legacy block ─────────────────────────────────────────────

        # ── Transformer-based receding-horizon actor ─────
        # Replaces the GNN actor for data collection.  Training update logic
        # (losses, QP labelling) is unchanged in this iteration — those are
        # updated in a follow-up.
        #
        # Sequence-length convention (must match CausalTransformerPolicy):
        #   history_len  = env.history_len      (e.g. 16)
        #   T            = history_len - 1       (e.g. 15)  — past steps in context
        #   obs_seq len  = T + 1 = history_len   (e.g. 16)
        #   action_seq   = T     = history_len-1  (e.g. 15)
        #   max_seq_len  = 2*T + 1               (e.g. 31)  — transformer sequence
        history_len = env.history_len
        tf_cfg = TransformerPolicyConfig(
            hidden_dim=128,
            num_heads=2,
            num_layers=1,
            obs_dim=env.obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            max_seq_len=2 * (history_len - 1) + 1,
            mlp_ratio=4.0,
            dropout_rate=0.0,   # disabled — see notes: JIT-baked self.key made it effectively static anyway, and weight_decay covers regularization for this size of transformer
            activation="gelu",
            action_head_hidden_dim=64,
            action_head_num_layers=2,
            dynamics_head_hidden_dim=128,
            dynamics_head_num_layers=3,
        )
        self.tf_policy = CausalTransformerPolicy(config=tf_cfg)

        tf_key, key = jr.split(key)
        # Dummy inputs for parameter initialisation — shapes only matter.
        dummy_obs    = jnp.zeros((n_agents, history_len,     env.obs_dim))
        dummy_act    = jnp.zeros((n_agents, history_len - 1, action_dim))
        dummy_u_nom  = jnp.zeros((n_agents, action_dim))
        tf_actor_params = self.tf_policy.init(tf_key, dummy_obs, dummy_act, dummy_u_nom)

        # total_steps × inner_epoch = total gradient updates — used for cosine decay.
        # If not provided, fall back to a constant LR (same as before).
        total_steps = kwargs.get('total_steps', None)
        inner_epoch  = inner_epoch  # already in scope from __init__ signature

        def _make_schedule(lr: float) -> optax.Schedule:
            if total_steps is not None:
                decay_steps = max(total_steps * inner_epoch, 1)
                return optax.cosine_decay_schedule(
                    init_value=lr, decay_steps=decay_steps, alpha=0.1
                )
            return lr  # constant

        # ── Frozen-backbone ablation (paper Table 9) ─────────────────────────
        # When True, only the actor head is trained in Phase 2; the transformer
        # backbone and dynamics head keep their Phase-1 pretrained weights.
        # Implemented as an optax.multi_transform routing every non-actor subtree
        # to set_to_zero, so the update is exactly zero rather than merely small.
        # Defaults to False -> the optimizer is then identical to the unablated
        # one, so normal training is bit-for-bit unaffected.
        self.freeze_backbone = kwargs.get('freeze_backbone', False)

        def _tf_param_labels(params):
            # Label every leaf 'train' iff it lives under the action_head
            # subtree, else 'frozen'.  Matches CausalTransformerPolicy.setup(),
            # which registers submodules as 'transformer', 'action_head',
            # 'dynamics_head' under params['params'].
            def _group(top_key: str) -> str:
                return 'train' if top_key == 'action_head' else 'frozen'
            return {'params': {
                k: jtu.tree_map(lambda _, kk=k: _group(kk), v)
                for k, v in params['params'].items()
            }}

        if self.freeze_backbone:
            tf_actor_optim = optax.multi_transform(
                {
                    'train': optax.adamw(
                        learning_rate=_make_schedule(lr_actor), weight_decay=1e-3
                    ),
                    'frozen': optax.set_to_zero(),
                },
                _tf_param_labels,
            )
        else:
            tf_actor_optim = optax.adamw(learning_rate=_make_schedule(lr_actor), weight_decay=1e-3)
        self.tf_actor_optim = optax.apply_if_finite(tf_actor_optim, 1_000_000)

        # ── Dynamics-head output scale (de-normalisation) ────────────────────
        # Some pretrained dynamics heads (the CrazyFlie checkpoint in particular)
        # were trained to predict the NORMALISED increment  delta_x / scale
        # rather than the raw delta_x.  Wherever state is advanced through the
        # dyn head,
        #     x_{k+1} = x_k + delta_x_hat * dyn_scale
        # this per-dimension scale restores physical units; without it the CBF
        # rollout barely moves and the horizon constraint is meaningless.
        # DI and DubinsCar heads output raw deltas, so the default (all-ones) is
        # an exact no-op for them.  Set via --dyn-scale-path.
        dyn_scale_path = kwargs.get('dyn_scale_path', None)
        if dyn_scale_path is not None and os.path.exists(dyn_scale_path):
            _scale = np.load(dyn_scale_path)['scale'].astype(np.float32)
            assert _scale.shape == (state_dim,), (
                f"dyn_scale shape {_scale.shape} != (state_dim={state_dim},)"
            )
            self._dyn_scale = jnp.asarray(_scale)
            print(f"  [BarrierFormer] Loaded dyn_scale from {dyn_scale_path} "
                  f"(min={_scale.min():.4g}, max={_scale.max():.4g})")
        else:
            self._dyn_scale = jnp.ones((state_dim,), dtype=jnp.float32)
            if dyn_scale_path is not None:
                print(f"  [BarrierFormer] WARNING: dyn_scale_path {dyn_scale_path} "
                      "not found — using unit scale (raw-delta dyn head).")
        self.tf_actor_train_state = TrainState.create(
            apply_fn=self.tf_policy.apply,
            params=tf_actor_params,
            tx=self.tf_actor_optim,
        )

        # Extra hyper-parameters accepted via **kwargs so the existing
        # GCBF constructor signature is unchanged.
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
        # Target network for label stability (updated via slow Polyak average).
        self.cbf_mlp_tgt = TrainState.create(
            apply_fn=self.cbf_mlp.apply,
            params=tree_copy(cbf_mlp_params),
            tx=empty_grad_tx(),
        )

        # set up key
        self.key = key
        self.buffer = MaskedReplayBuffer(size=buffer_size)
        self.unsafe_buffer = MaskedReplayBuffer(size=buffer_size // 2)
        self.rng = np.random.default_rng(seed=seed + 1)

    @property
    def config(self) -> dict:
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
            'freeze_backbone': self.freeze_backbone,
        }

    @property
    def tf_actor_params(self) -> dict:
        """Frozen parameter pytree for the CausalTransformerPolicy.

        Passed as the ``params`` argument to ``rollout_transformer`` so that
        ``jax.jit`` can treat them as a dynamic JAX pytree (not static).
        """
        return self.tf_actor_train_state.params

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, unsafe_mask: Array) -> jnp.ndarray:
        # safe if in the labeling_horizon, the agent is always safe.
        # labeling_horizon is decoupled from the transformer rollout horizon so
        # that the label density matches the original GNN paper (default 32)
        # even when the predictive horizon is shorter (e.g. 6).
        def safe_rollout(single_rollout_mask: Array) -> Array:
            safe_rollout_mask = jnp.ones_like(single_rollout_mask)
            for i in range(single_rollout_mask.shape[0]):
                start = 0 if i < self.labeling_horizon else i - self.labeling_horizon
                safe_rollout_mask = safe_rollout_mask.at[start: i + 1].set(
                    ((1 - single_rollout_mask[i]) * safe_rollout_mask[start: i + 1]).astype(jnp.bool_))
                # initial state is always safe
                safe_rollout_mask = safe_rollout_mask.at[0].set(1)
            return safe_rollout_mask

        safe = jax_vmap(jax_vmap(safe_rollout, in_axes=1, out_axes=1))(unsafe_mask)
        return safe

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        if params is None:
            params = self.actor_train_state.params
        action = 2 * self.actor.get_action(params, graph) + self._env.u_ref(graph)
        return action

    def act_transformer(
            self,
            graph: GraphsTuple,
            params: Optional[Params] = None,
    ) -> Action:
        """Single-step action from the transformer actor head.

        Transposes history tensors from the env storage convention
        ``[T, n_agents, dim]`` to the transformer input convention
        ``[n_agents, T, dim]``, then runs one forward pass and returns
        ``u_applied = u_nom + delta_u``.

        Parameters
        ----------
        graph  : Single (unbatched) environment graph.
        params : Transformer parameter pytree.  Defaults to the current
                 training weights.
        """
        if params is None:
            params = self.tf_actor_train_state.params
        # env stores history as [T, n_agents, dim]; transformer expects [n_agents, T, dim]
        obs_seq    = graph.env_states.obs_history.swapaxes(0, 1)     # [n_agents, T+1, obs_dim]
        action_seq = graph.env_states.action_history.swapaxes(0, 1)  # [n_agents, T,   act_dim]
        u_nom      = self._env.u_ref(graph)                           # [n_agents, act_dim]
        out        = self.tf_policy.apply(params, obs_seq, action_seq, u_nom, True)
        return out.u_applied   # [n_agents, act_dim]

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Action:
        if params is None:
            params = self.actor_params
        action = self.actor_train_state.apply_fn(params, graph, key)
        return 2 * action + self._env.u_ref(graph)

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnums=1)
    def update_tgt(self, cbf_tgt: TrainState, cbf: TrainState, tau: float) -> TrainState:
        tgt_params = optax.incremental_update(cbf.params, cbf_tgt.params, tau)
        return cbf_tgt.replace(params=tgt_params)

    @ft.partial(jax.jit, static_argnums=(0,))
    def get_b_u_qp(self, b_graph: GraphsTuple, params) -> Action:
        b_u_qp, bT_relaxation = jax_vmap(ft.partial(self.get_qp_action, cbf_params=params))(b_graph)
        return b_u_qp

    @ft.partial(jax.jit, static_argnums=(0, 2, 4, 5))   # self, cbf_apply_fn, horizon, n_sqp_iter
    def get_b_u_sqp(
            self,
            b_graph: GraphsTuple,
            cbf_apply_fn: Callable,               # static: Python callable, not a JAX array
            params,
            horizon: int = 5,
            n_sqp_iter: int = 25,
            relax_penalty: float = 1e3,
            gamma: float = 0.0,
    ) -> tuple[Action, Array, Array, Array]:
        """Batch SQP action labels — mirrors ``get_b_u_qp`` structure exactly.

        Uses ``jax_vmap`` over the chunk (same as get_b_u_qp).  Memory is
        controlled at the call site by passing pre-chunked graphs — see the
        chunked-loop pattern in ``update_nets``.

        ``cbf_apply_fn`` is marked ``static_argnums=(0, 2)`` so JAX sees it as
        a compile-time constant (a Python callable cannot be a JAX tracer).
        Since the same function is passed every call, JIT compiles once.

        Parameters
        ----------
        cbf_apply_fn : Callable ``(params, obs: [n_agents, obs_dim]) → [n_agents, 1]``
                       CBFMLP apply function — pass e.g. ``cbf_mlp_model.apply``.

        Returns
        -------
        b_u_sqp            : [B, n_agents, 2]   actor head labels (first step only)
        b_violation_before : [B, H, n_agents]   DTCBF violation at ΔU=0 (before SQP)
        b_violation_after  : [B, H, n_agents]   DTCBF violation at ΔU_final (after SQP)
        b_n_iter_used      : [B]                actual SQP iterations per sample
        """
        b_u_sqp, b_violation_before, b_violation_after, b_n_iter_used = jax_vmap(
            ft.partial(
                self.get_sqp_action,
                cbf_apply_fn=cbf_apply_fn,
                horizon=horizon,
                n_sqp_iter=n_sqp_iter,
                relax_penalty=relax_penalty,
                cbf_params=params,
                gamma=gamma,
            )
        )(b_graph)
        return b_u_sqp, b_violation_before, b_violation_after, b_n_iter_used


    '''old fuction with chunking

    def update_nets(self, rollout: Rollout, safe_mask, unsafe_mask):
        update_info = {}

        # ── Label collection ─────────
        n_chunks   = 8
        batch_size = len(rollout.graph.states)
        chunk_size = batch_size // n_chunks

        # ── LEGACY GNN path (commented out) ──────────────
        # The GNN single-step QP labels are no longer used for training once
        # the transformer + CBFMLP path is active.  Kept here for easy
        # ablation / rollback — uncomment and swap ``batch_orig`` below.
        #
        # b_u_qp = []
        # for ii in range(n_chunks):
        #     graph = jtu.tree_map(
        #         lambda x: x[ii * chunk_size: (ii + 1) * chunk_size], rollout.graph
        #     )
        #     b_u_qp.append(jax2np(self.get_b_u_qp(graph, self.cbf_tgt.params)))
        # b_u_qp = tree_merge(b_u_qp)
        # batch_orig = Batch(rollout.graph, safe_mask, unsafe_mask, b_u_qp)
        # ... then call update_inner(self.cbf_train_state, self.actor_train_state, batch) ...

        # ── Transformer path: SQP horizon labels ─────────
        # Each chunk is JIT-compiled once (cbf_apply_fn is static).
        # Memory per chunk = chunk_size * (SQP vars).
        cbf_apply_fn = self.cbf_mlp.apply    # CBFMLP from nn/cbf.py — static callable
        b_u_sqp, b_r_sqp, b_x_hat = [], [], []
        for ii in range(n_chunks):
            graph = jtu.tree_map(
                lambda x: x[ii * chunk_size: (ii + 1) * chunk_size],
                rollout.graph,
            )
            chunk = jax2np(
                self.get_b_u_sqp(
                    graph,
                    cbf_apply_fn,
                    self.cbf_mlp_tgt.params,           # target params for label stability
                    horizon=self.sqp_horizon_len,
                    n_sqp_iter=self.n_sqp_iter,
                    relax_penalty=self.relax_penalty,
                )
            )
            b_u_sqp.append(chunk[0])   # [chunk, n_agents, 2]
            b_r_sqp.append(chunk[1])   # [chunk, n_agents]
            b_x_hat.append(chunk[2])   # [chunk, H+1, n_agents, 4]

        b_u_sqp = tree_merge(b_u_sqp)   # [B, n_agents, 2]
        b_x_hat = tree_merge(b_x_hat)   # [B, H+1, n_agents, 4]
        b_r_sqp_arr = tree_merge(b_r_sqp)  # [B, n_agents]
        self._sqp_slack_mean = float(jnp.mean(b_r_sqp_arr))
        self._sqp_slack_max  = float(jnp.max(b_r_sqp_arr))

        # The chunked SQP loop processes exactly n_chunks * chunk_size samples.
        # If total batch size is not divisible by n_chunks, the remainder is
        # silently dropped here so that all Batch fields have matching leading dim.
        n_processed = n_chunks * chunk_size
        graph_trim     = jtu.tree_map(lambda x: x[:n_processed], rollout.graph)
        safe_mask_trim   = safe_mask[:n_processed]
        unsafe_mask_trim = unsafe_mask[:n_processed]

        # u_qp field carries the SQP actor-head label; x_hat_horizon carries
        # the SQP-computed safe trajectory for CBFMLP horizon supervision.
        batch_orig = Batch(
            graph_trim, safe_mask_trim, unsafe_mask_trim,
            b_u_sqp,    # actor head label (same field name, different source)
            b_x_hat,    # horizon states for CBFMLP training
        )

        # Guard: update_inner_transformer requires x_hat_horizon to be a
        # concrete array.  This assertion fires at Python level (before JIT),
        # so the error message is clear.  Inside lax.scan a None value would
        # silently produce a TypeError from jax.vmap over None.
        assert batch_orig.x_hat_horizon is not None, (
            "update_inner_transformer requires x_hat_horizon to be populated. "
            "Ensure the SQP path in update_nets ran successfully before this call."
        )

        for i_epoch in range(self.inner_epoch):
            # Use n_processed (= n_chunks * chunk_size) — the size all batch_orig
            # fields share after the SQP trim step above, NOT rollout.length which
            # may be slightly larger due to non-divisible buffer merge sizes.
            n_batches = n_processed // self.batch_size
            if n_batches == 0:
                break  # not enough data for even one minibatch
            idx       = self.rng.choice(n_processed, size=n_batches * self.batch_size, replace=False)
            batch_idx = idx.reshape(n_batches, self.batch_size)
            batch     = jtu.tree_map(lambda x: x[batch_idx], batch_orig)

            cbf_mlp_ts, tf_actor_ts, update_info = self.update_inner_transformer(
                self.cbf_mlp_train_state, self.tf_actor_train_state, batch
            )
            self.cbf_mlp_train_state  = cbf_mlp_ts
            self.tf_actor_train_state = tf_actor_ts

        # Slow Polyak update of the CBFMLP target network.
        self.cbf_mlp_tgt = self.update_tgt(self.cbf_mlp_tgt, self.cbf_mlp_train_state, self.cbf_tgt_tau)

        return update_info'''
    def update_nets(self, rollout: Rollout, safe_mask, unsafe_mask):
        update_info = {}


        cbf_apply_fn = self.cbf_mlp.apply
        b_u_sqp, b_violation_before, b_violation_after, b_n_iter_used = self.get_b_u_sqp(
            rollout.graph,
            cbf_apply_fn,
            self.cbf_mlp_tgt.params,
            horizon=self.sqp_horizon_len,
            n_sqp_iter=self.n_sqp_iter,
            relax_penalty=self.relax_penalty,
            gamma=self.gamma,
        )
        # b_u_sqp           : [B, n_agents, 2]  — actor head labels
        # b_violation_before: [B, H, n_agents]  — DTCBF violation at ΔU=0 (before SQP)
        # b_violation_after : [B, H, n_agents]  — DTCBF violation at ΔU_final (after SQP)
        # b_n_iter_used     : [B]               — actual SQP iterations (early stopping)

        # ── NaN / Inf guard — diagnose silent apply_if_finite skips ──────────
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


        # Single threshold for "SQP solved this sample" — reused below for the
        # reported converged fraction so the metric and the gating mask agree.
        sqp_conv_thresh = 0.05
        sqp_converged = (b_violation_after.max(axis=(-2, -1)) < sqp_conv_thresh)   # [B] bool

        batch_orig = Batch(rollout.graph, safe_mask_trim, unsafe_mask_trim, b_u_sqp, sqp_converged)

        for i_epoch in range(self.inner_epoch):
            # Use n_processed (= n_chunks * chunk_size) — the size all batch_orig
            # fields share after the SQP trim step above, NOT rollout.length which
            # may be slightly larger due to non-divisible buffer merge sizes.
            n_batches = n_processed // self.batch_size
            if n_batches == 0:
                break  # not enough data for even one minibatch
            idx       = self.rng.choice(n_processed, size=n_batches * self.batch_size, replace=False)
            batch_idx = idx.reshape(n_batches, self.batch_size)
            batch     = jtu.tree_map(lambda x: x[batch_idx], batch_orig)

            cbf_mlp_ts, tf_actor_ts, update_info = self.update_inner_transformer(
                self.cbf_mlp_train_state, self.tf_actor_train_state, batch
            )
            self.cbf_mlp_train_state  = cbf_mlp_ts
            self.tf_actor_train_state = tf_actor_ts

        # Slow Polyak update of the CBFMLP target network.
        self.cbf_mlp_tgt = self.update_tgt(self.cbf_mlp_tgt, self.cbf_mlp_train_state, self.cbf_tgt_tau)


        update_info['sqp/violation_before_max']  = float(b_violation_before.max())
        update_info['sqp/violation_before_mean'] = float(b_violation_before.mean())
        update_info['sqp/violation_after_max']   = float(b_violation_after.max())
        update_info['sqp/violation_after_mean']  = float(b_violation_after.mean())
        update_info['sqp/n_iter_used_mean']      = float(b_n_iter_used.mean())
        update_info['sqp/converged_frac']        = float(
            jnp.mean(b_violation_after.max(axis=(-2, -1)) < sqp_conv_thresh)
        )
        return update_info

    # ------------------------------------------------------------------
    # Checkpoint: save / load transformer + CBFMLP weights.
    # File layout matches the GNN convention (save_dir/step/filename.pkl)
    # so existing tooling and eval scripts work unchanged.
    #
    # Files written per checkpoint step:
    #   actor.pkl      — transformer (actor head + dynamics head) params +
    #                    Adam opt_state + step counter
    #   cbf.pkl        — CBFMLP (train network) params + opt_state + step
    #   cbf_tgt.pkl    — CBFMLP target network params only (no optimizer)
    #
    # opt_state is stored alongside params so training resumes without a
    # learning-rate spike from re-warmed Adam moments.
    # GNN actor / CBF weights are intentionally NOT saved.
    # ------------------------------------------------------------------

    @staticmethod
    def _ts_to_dict(ts) -> dict:
        """Picklable subset of a TrainState (no tx closures)."""
        return {'params': ts.params, 'opt_state': ts.opt_state, 'step': ts.step}

    def save(self, save_dir: str, step: int):
        model_dir = os.path.join(save_dir, str(step))
        os.makedirs(model_dir, exist_ok=True)

        # actor.pkl — transformer: actor head (delta_u) + dynamics head (delta_x_hat)
        pickle.dump(
            self._ts_to_dict(self.tf_actor_train_state),
            open(os.path.join(model_dir, 'actor.pkl'), 'wb'),
        )
        # cbf.pkl — CBFMLP train network
        pickle.dump(
            self._ts_to_dict(self.cbf_mlp_train_state),
            open(os.path.join(model_dir, 'cbf.pkl'), 'wb'),
        )
        # cbf_tgt.pkl — CBFMLP target network (Polyak copy, no optimizer)
        pickle.dump(
            self.cbf_mlp_tgt.params,
            open(os.path.join(model_dir, 'cbf_tgt.pkl'), 'wb'),
        )

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))

        d = pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb'))
        self.tf_actor_train_state = self.tf_actor_train_state.replace(
            params=d['params'], opt_state=d['opt_state'], step=d['step']
        )

        d = pickle.load(open(os.path.join(path, 'cbf.pkl'), 'rb'))
        self.cbf_mlp_train_state = self.cbf_mlp_train_state.replace(
            params=d['params'], opt_state=d['opt_state'], step=d['step']
        )

        # cbf_tgt.pkl may be absent in older checkpoints — fall back to
        # syncing target with the live CBFMLP weights.
        tgt_path = os.path.join(path, 'cbf_tgt.pkl')
        tgt_params = (
            pickle.load(open(tgt_path, 'rb'))
            if os.path.exists(tgt_path)
            else self.cbf_mlp_train_state.params
        )
        self.cbf_mlp_tgt = self.cbf_mlp_tgt.replace(params=tgt_params)


    def sample_batch(self, rollout: Rollout, safe_mask, unsafe_mask):
        self.buffer.append(rollout, safe_mask, unsafe_mask)
        # Select EPISODES that contain at least one unsafe agent-timestep.
        # unsafe_mask: (n_env, T, n_agents) → max over T and n_agents → (n_env,)
        unsafe_ep_mask = unsafe_mask.max(axis=-1).max(axis=-1).astype(bool)
        if unsafe_ep_mask.any():
            self.unsafe_buffer.append(
                jtu.tree_map(lambda x: x[unsafe_ep_mask], rollout),
                safe_mask[unsafe_ep_mask],
                unsafe_mask[unsafe_ep_mask],
            )

        # Flatten current episode to individual transitions: (n_env, T, ...) → (n_env*T, ...)
        flat_rollout = jtu.tree_map(lambda x: merge01(x), rollout)
        flat_safe    = merge01(safe_mask)
        flat_unsafe  = merge01(unsafe_mask)

        if self.buffer.length > self.batch_size:
            memory, safe_mem, unsafe_mem = self.buffer.sample(rollout.length)

            if self.unsafe_buffer.length > 0:
                try:
                    unsafe_memory, safe_unsafe_mem, unsafe_unsafe_mem = \
                        self.unsafe_buffer.sample(rollout.length * rollout.time_horizon)
                except (ValueError, AttributeError):
                    unsafe_memory     = flat_rollout
                    safe_unsafe_mem   = flat_safe
                    unsafe_unsafe_mem = flat_unsafe
            else:
                unsafe_memory     = flat_rollout
                safe_unsafe_mem   = flat_safe
                unsafe_unsafe_mem = flat_unsafe

            combined          = tree_merge([memory, flat_rollout])
            combined_safe     = tree_merge([safe_mem, flat_safe])
            combined_unsafe   = tree_merge([unsafe_mem, flat_unsafe])

            rollout_batch     = tree_merge([unsafe_memory, combined])
            safe_mask_batch   = tree_merge([safe_unsafe_mem, combined_safe])
            unsafe_mask_batch = tree_merge([unsafe_unsafe_mem, combined_unsafe])
        else:
            rollout_batch     = flat_rollout
            safe_mask_batch   = flat_safe
            unsafe_mask_batch = flat_unsafe

        return rollout_batch, safe_mask_batch, unsafe_mask_batch

    def update(self, rollout: Rollout, step: int) -> dict:
        key, self.key = jr.split(self.key)

        # (n_collect, T)
        unsafe_mask = jax_vmap(jax_vmap(self._env.unsafe_mask))(rollout.graph)
        safe_mask = self.safe_mask(unsafe_mask)
        safe_mask, unsafe_mask = jax2np(safe_mask), jax2np(unsafe_mask)

        rollout_np = jax2np(rollout)
        del rollout
        rollout_batch, safe_mask_batch, unsafe_mask_batch = self.sample_batch(rollout_np, safe_mask, unsafe_mask)

        # inner loop
        update_info = self.update_nets(rollout_batch, safe_mask_batch, unsafe_mask_batch)

        return update_info

    # def get_qp_action(
    #         self,
    #         graph: GraphsTuple,
    #         relax_penalty: float = 1e3,
    #         cbf_params=None,
    #         qp_settings: JaxProxQP.Settings = None,
    # ) -> [Action, Array]:
    #     assert graph.is_single  # consider single graph
    #     agent_node_mask = graph.node_type == 0
    #     agent_node_id = mask2index(agent_node_mask, self.n_agents)

    #     def h_aug(new_agent_state: State) -> Array:
    #         new_state = graph.states.at[agent_node_id].set(new_agent_state)
    #         new_graph = self._env.add_edge_feats(graph, new_state)
    #         return self.get_cbf(new_graph, params=cbf_params)

    #     agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)
    #     h = h_aug(agent_state).squeeze(-1)
    #     h_x = jax.jacobian(h_aug)(agent_state).squeeze(1)

    #     dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)
    #     Lf_h = ei.einsum(h_x, dyn_f, "agent_i agent_j nx, agent_j nx -> agent_i")
    #     Lg_h = ei.einsum(h_x, dyn_g, "agent_i agent_j nx, agent_j nx nu -> agent_i agent_j nu")
    #     Lg_h = Lg_h.reshape((self.n_agents, -1))

    #     u_lb, u_ub = self._env.action_lim()
    #     u_lb = u_lb[None, :].repeat(self.n_agents, axis=0).reshape(-1)
    #     u_ub = u_ub[None, :].repeat(self.n_agents, axis=0).reshape(-1)
    #     u_ref = self._env.u_ref(graph).reshape(-1)

    #     # construct QP: min x^T H x + g^T x, s.t. Cx <= b
    #     H = jnp.eye(self._env.action_dim * self.n_agents + self.n_agents, dtype=jnp.float32)
    #     H = H.at[-self.n_agents:, -self.n_agents:].set(H[-self.n_agents:, -self.n_agents:] * 10.0)
    #     g = jnp.concatenate([-u_ref, relax_penalty * jnp.ones(self.n_agents)])
    #     C = -jnp.concatenate([Lg_h, jnp.eye(self.n_agents)], axis=1)
    #     b = Lf_h + self.alpha * 0.1 * h

    #     r_lb = jnp.array([0.] * self.n_agents, dtype=jnp.float32)
    #     r_ub = jnp.array([jnp.inf] * self.n_agents, dtype=jnp.float32)
    #     l_box = jnp.concatenate([u_lb, r_lb], axis=0)
    #     u_box = jnp.concatenate([u_ub, r_ub], axis=0)

    #     qp = JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)
    #     if qp_settings is None:
    #         qp_settings = JaxProxQP.Settings.default()
    #     qp_settings.dua_gap_thresh_abs = None
    #     solver = JaxProxQP(qp, qp_settings)
    #     sol = solver.solve()

    #     assert sol.x.shape == (self.action_dim * self.n_agents + self.n_agents,)
    #     u_opt, r = sol.x[:self.action_dim * self.n_agents], sol.x[-self.n_agents:]
    #     u_opt = u_opt.reshape(self.n_agents, -1)

    #     return u_opt, r

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
    ) -> tuple[Action, Array, Array, Array]:

        if alpha is None:
            alpha = self.alpha
        assert graph.is_single

        # ── Setup 
        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)# agent_state: [n_agents, 4]
        goal_states = graph.type_states(type_idx=1, n_type=self.n_agents) # goal_states: [n_agents, 4]  — fixed throughout the horizon
        obstacles   = graph.env_states.obstacle # obstacles: Rectangle pytree — world-frame, fixed throughout the horizon

        u_nom_t    = self._env.u_ref(graph)                          # [n_agents, 2]

        u_lb_scalar, u_ub_scalar = self._env.action_lim()           # [action_dim] each
        u_lb_tiled = jnp.tile(u_lb_scalar, horizon * self.n_agents) # [H*n_agents*2]
        u_ub_tiled = jnp.tile(u_ub_scalar, horizon * self.n_agents) # [H*n_agents*2]

        # Problem dimensions (Python ints — static at JIT trace time)
        n_u = horizon * self.n_agents * self._env.action_dim         # H * n_agents * 2
        n_r = horizon * self.n_agents                                # H * n_agents (slacks)
        n_c = n_r                                                    # H * n_agents (constraints)

        # Initial correction, slack, and placeholder horizon states
        delta_U = jnp.zeros((horizon, self.n_agents, self._env.action_dim), dtype=jnp.float32)                                                              # [H, n_agents, 2]
        r_init  = jnp.zeros(n_r, dtype=jnp.float32)                  # [H*n_agents]

        # QP settings — created once outside the loop (Python-level object)
        if qp_settings is None:
            qp_settings = JaxProxQP.Settings.default()
        qp_settings.dua_gap_thresh_abs = None

        obs_from_state: Callable = make_obs_from_state_fn(
            goal_states=goal_states,
            obstacles=obstacles,
            n_rays=self._env._params["n_rays"],
            comm_radius=self._env._params["comm_radius"],
            num_agents=self.n_agents,
            # Delegate to the env's own observation builder so the reconstructed
            # o_hat matches training exactly and supports 3-D envs.  Verified
            # bit-identical to the inline fallback on DI and DubinsCar; the
            # fallback is 2-D-only and raises on CrazyFlie (state 12 + goal 3 +
            # n_rays*5), which is why CF previously needed a separate module.
            env=self._env,
        )

        def h_from_state(x_k: Array) -> Array:
            o_k = obs_from_state(x_k)                              # [n_agents, obs_dim]
            h_k = cbf_apply_fn(cbf_params, o_k)                   # [n_agents, 1]
            return h_k.reshape(self.n_agents)                      # [n_agents]

        # ── c_of_delta_U: constraint residuals + horizon states ──────────────
        def c_of_delta_U(delta_U_in: Array) -> tuple[Array, Array]:

            def scan_fn(x_carry, delta_u_k):
                # x_carry:    [n_agents, 4]
                # delta_u_k:  [n_agents, 2]
                u_nom_k   = self._env.u_nom_from_state(x_carry, goal_states)
                u_applied = u_nom_k + delta_u_k              # [n_agents, 2]
                # Straight-through clipped Euler step.
                #   forward  = clip_state(x + increment) -> reachable states, so the
                #              CBF eval, the DTCBF residual c, and the feasibility
                #              test all use dynamics the env actually follows (and
                #              consistent with the clipped CBF-horizon rollout).
                #   backward = unclipped -> the SQP Jacobian dc/dU keeps a non-
                #              degenerate gradient at v=+/-v_max, so the QP can still
                #              plan braking (the reason step_simulator was left
                #              unclipped in the first place).
                # Scoped to the SQP only: step_simulator stays the UNCLIPPED
                # increment for the loss_dyn target (the dyn head must predict the
                # raw increment; the rollout clips afterward).
                x_next_raw = self._env.step_simulator(x_carry, u_applied)   # unclipped
                x_next     = x_next_raw + jax.lax.stop_gradient(
                    self._env.clip_state(x_next_raw) - x_next_raw
                )
                return x_next, x_next

            _, x_hat_steps = jax.lax.scan(
                scan_fn,
                agent_state,    # [n_agents, 4] — initial state
                delta_U_in,     # [H, n_agents, 2] — xs scanned over axis-0
            )
            # x_hat_steps: [H, n_agents, 4]  (states at t+1 … t+H)

            x_hat_all = jnp.concatenate(
                [agent_state[None], x_hat_steps], axis=0
            )                                                # [H+1, n_agents, 4]

            # CBFMLP with full obs (lidar re-cast + rel-goal) at each x̂_k
            h_vals = jax.vmap(h_from_state)(x_hat_all)      # [H+1, n_agents]

            # Discrete-time CBF residual: c_k ≤ 0 when constraint satisfied
            c = gamma - h_vals[1:] + (1.0 - alpha) * h_vals[:-1]
            # c: [H, n_agents]

            return c, x_hat_all

        # ── Compute violation BEFORE any correction (diagnostic baseline) ──────
        # delta_U = zeros at this point so c_init = CBF residuals under u_nom only.
        c_init, x_hat_init_actual = c_of_delta_U(delta_U)
        violation_before = jnp.maximum(c_init, 0.0)          # [H, n_agents]


        def body_fn(carry: tuple) -> tuple:
            delta_U_i, _, x_hat_all, i, c_nom = carry

            c_nom_flat = c_nom.reshape(-1)                   # [n_c]

            # Linearise: Jacobian dc/d(delta_U).
            # step_simulator is JAX-differentiable (pure Euler integration).
            A_cbf_raw = jax.jacobian(
                lambda dU: c_of_delta_U(dU)[0]
            )(delta_U_i)
            # A_cbf_raw: [H, n_agents, H, n_agents, 2]
            A_cbf = A_cbf_raw.reshape(n_c, n_u)             # [n_c, n_u]

            # Build QP — decision variable: x = [delta_flat | xi]
            # Slack cost is purely quadratic (H_slack = relax_penalty, g_slack = 0).
            # A linear slack term relax_penalty*1^T xi causes ProxQP to require dual
            # variables of magnitude relax_penalty; at 10k+ this overflows float32.
            # Quadratic penalty keeps duals O(1) and is well-conditioned for any
            # relax_penalty value.
            H_mat = jnp.eye(n_u + n_r, dtype=jnp.float32)
            H_mat = H_mat.at[-n_r:, -n_r:].set(
                H_mat[-n_r:, -n_r:] * relax_penalty
            )                                                # [n_u+n_r, n_u+n_r]

            g_vec = jnp.concatenate([
                delta_U_i.reshape(-1).astype(jnp.float32),   # [n_u]  +ΔU^(i)
                jnp.zeros(n_r, dtype=jnp.float32),            # [n_r]  quadratic only
            ])                                               # [n_u+n_r]

            # CBF constraint: c_nom + A_cbf @ δ ≤ ξ  →  [A_cbf | −I][δ; ξ] ≤ −c_nom
            C_mat = jnp.concatenate(
                [A_cbf, -jnp.eye(n_c, dtype=jnp.float32)], axis=1
            )                                                # [n_c, n_u+n_r]
            b_vec = -c_nom_flat.astype(jnp.float32)         # [n_c]

            u_nom_traj = jax.vmap(
                lambda x_k: self._env.u_nom_from_state(x_k, goal_states)
            )(x_hat_all[:-1])                               # [H, n_agents, 2]
            u_nom_traj_flat = u_nom_traj.reshape(-1).astype(jnp.float32)  # [n_u]
            delta_lb = (u_lb_tiled - u_nom_traj_flat - delta_U_i.reshape(-1)).astype(jnp.float32)                    # [n_u]
            delta_ub = (u_ub_tiled - u_nom_traj_flat - delta_U_i.reshape(-1)).astype(jnp.float32)                    # [n_u]
            l_box = jnp.concatenate([delta_lb, jnp.zeros(n_r, dtype=jnp.float32)])                                   # [n_u+n_r]
            u_box = jnp.concatenate([delta_ub, jnp.full(n_r, jnp.inf, dtype=jnp.float32)])                                               # [n_u+n_r]

            qp     = JaxProxQP.QPModel.create(H_mat, g_vec, C_mat, b_vec, l_box, u_box)
            solver = JaxProxQP(qp, qp_settings)
            sol    = solver.solve()

            delta_flat  = sol.x[:n_u]                       # [n_u]
            delta_U_new = delta_U_i + delta_flat.reshape(horizon, self.n_agents, self._env.action_dim)                                               # [H, n_agents, 2]

            c_new, x_hat_new = c_of_delta_U(delta_U_new)   # [H,n_a], [H+1,n_a,4]

            return (delta_U_new, r_init, x_hat_new, i + 1, c_new)

        def cond_fn(carry: tuple) -> bool:
            _, _, _, i, c = carry
            # Continue while budget not exhausted AND any constraint still violated.
            return jnp.logical_and(i < n_sqp_iter, jnp.any(c > 0.0))

        # ── Run while_loop with early stopping ──────────────────────────────────
        init_carry = (delta_U, r_init, x_hat_init_actual,
                      jnp.array(0, dtype=jnp.int32), c_init)
        final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        delta_U_final, _, _, n_iter_used, c_final = final_carry
        # delta_U_final : [H, n_agents, 2]  — optimal correction (or zero if already safe)
        # n_iter_used   : scalar int32       — actual SQP iterations executed
        # c_final       : [H, n_agents]      — DTCBF residuals at delta_U_final

        violation_after = jnp.maximum(c_final, 0.0)         # [H, n_agents]

        # Actor head label — first correction step only.
        # At k=0: x̄_0 = agent_state = x_t, so u_nom(x̄_0) = u_nom_t exactly.
        u_opt = u_nom_t + delta_U_final[0]                   # [n_agents, 2]
        u_opt = jnp.clip(u_opt, u_lb_scalar[None], u_ub_scalar[None])

        return u_opt, violation_before, violation_after, n_iter_used

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 2))
    def update_inner_transformer(
            self,
            cbf_mlp_train_state: TrainState,
            tf_actor_train_state: TrainState,
            batch: Batch,
    ) -> tuple[TrainState, TrainState, dict]:
        
        def update_fn(carry, minibatch: Batch):
            cbf_mlp, tf_actor, dropout_key = carry
            dropout_key, step_key = jr.split(dropout_key)

            safe_mask_flat   = merge01(minibatch.safe_mask)   # [mb*n_agents]
            unsafe_mask_flat = merge01(minibatch.unsafe_mask)
            # Current obs: last entry in the history window. obs_history stored as [mb, T+1, n_agents, obs_dim].
            obs_current = minibatch.graph.env_states.obs_history[:, -1, :, :]
            # Transpose history for the transformer: [mb, T+1, n_agents, d]→[mb, n_agents, T+1, d]
            obs_seq    = minibatch.graph.env_states.obs_history.transpose(0, 2, 1, 3)
            action_seq = minibatch.graph.env_states.action_history.transpose(0, 2, 1, 3)

            u_nom_batch = jax.vmap(self._env.u_ref)(minibatch.graph) # u_nom_batch: [mb, n_agents, act_dim]

            # Per-sample goal and obstacle for obs reconstruction.
            goal_batch     = minibatch.graph.env_states.goal      # [mb, n_agents, 4]
            obstacle_batch = minibatch.graph.env_states.obstacle   # [mb, obstacle pytree]

            def get_loss(cbf_mlp_params, tf_actor_params):
                labeled_mask      = jnp.logical_or(minibatch.safe_mask, minibatch.unsafe_mask)  # [mb, n_agents]
                labeled_mask_flat = merge01(labeled_mask)                  # [mb*n_agents]

                # Params with stop_gradient ONLY on the dynamics-head subtree.
                # Forward pass is identical (stop_gradient is identity in forward),
                # but no gradient flows back into θ_x.  Used everywhere the dyn head
                # is called by a CBF loss, so the dyn head behaves like a frozen
                # black-box simulator surrogate — trained ONLY by loss_dyn.
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
                    return self.cbf_mlp.apply(cbf_mlp_params, o).reshape(self.n_agents) # o: [n_agents, obs_dim] → [n_agents]

                def eval_cbf_stopped(o):
                    # Used only for unlabeled samples in the DTCBF loss — for those we
                    # don't want to push the CBF (no safe/unsafe ground truth) but the
                    # actor still receives gradient through state → obs.
                    return self.cbf_mlp.apply(jax.lax.stop_gradient(cbf_mlp_params), o).reshape(self.n_agents)

                h      = jax.vmap(eval_cbf)(obs_current)    # [mb, n_agents]
                h_flat = merge01(h)                         # [mb*n_agents]

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
                        tf_actor_params, obs_s, act_s, u_n, False,
                        rngs={'dropout': step_key},
                    )
                    return out.delta_u, out.u_applied, out.z_t


                delta_u_batch, u_applied_batch, z_t_batch = jax.vmap(tf_fwd)(
                    obs_seq, action_seq, u_nom_batch
                )


                u_sqp         = minibatch.u_qp                   # [mb, n_agents, act_dim]  (SQP label)
                delta_a_label = u_sqp - u_nom_batch              # [mb, n_agents, act_dim]  Δu*_0

                per_sample_loss = jnp.sum(jnp.square(delta_u_batch - delta_a_label), axis=-1)  # [mb, n_agents]
                per_sample_loss = per_sample_loss.mean(axis=-1)  # [mb]
                if minibatch.sqp_converged is not None:
                    conv_mask = minibatch.sqp_converged.astype(jnp.float32)  # [mb]
                    n_converged = jnp.maximum(conv_mask.sum(), 1.0)
                    loss_action = (per_sample_loss * conv_mask).sum() / n_converged
                else:
                    loss_action = per_sample_loss.mean()


                x_t = obs_current[:, :, :self._env.state_dim]   # [mb, n_agents, 4]
                u_lb_s, u_ub_s = self._env.action_lim()
                u_clipped = jnp.clip(u_applied_batch, u_lb_s[None, None, :], u_ub_s[None, None, :])  # [mb, n_agents, act_dim]

                x_next_true  = jax.vmap(lambda xi, ui: self._env.step_simulator(xi, ui))(x_t, u_clipped)  # [mb, n_agents, 4]
                delta_x_true = x_next_true - x_t                  # [mb, n_agents, 4]

                # L_dyn updates θ_T (transformer) AND θ_x (dyn head) — paper Sec. 4.6
                # writes the loss as L_dyn(θ_T, θ_x).  We only need to stop the
                # gradient that would flow through the ACTOR head (via u_applied =
                # u_nom + delta_u), so L_dyn does NOT pull θ_u.  z_t stays live so
                # gradient still flows to the transformer backbone.
                u_applied_sg = jax.lax.stop_gradient(u_clipped)

                def dyn_only(z, u):
                    return self.tf_policy.apply(
                        tf_actor_params, z, u,
                        method=lambda mdl, zz, uu: mdl.call_dynamics_head(zz, uu),
                    )

                delta_x_hat_for_dyn = jax.vmap(dyn_only)(z_t_batch, u_applied_sg)
                # The dyn head is supervised in NORMALISED units, so the target is
                # divided by dyn_scale to match (no-op when dyn_scale == 1).
                delta_x_true_norm = delta_x_true / self._dyn_scale[None, None, :]
                loss_dyn     = jnp.mean(jnp.sum(jnp.square(delta_x_hat_for_dyn - delta_x_true_norm), axis=-1))


                # Single-step CBF supervision.
                # x̂_1 is produced by the DYNAMICS HEAD (not the simulator) so the
                # training-time gradient pathway matches deployment exactly:
                #   actor → u_applied → dyn head → x̂_1 → obs_next → CBF → loss
                # CBF gradient flows on labeled samples (per GCBF+ DTCBF loss); on
                # unlabeled samples we keep stop_gradient on CBF params.
                # Dyn-head params are stop_gradient'd (tf_params_dyn_sg) so the CBF
                # loss does NOT update θ_x — the dyn head is trained only by L_dyn
                # and acts as a frozen black-box simulator surrogate here.
                delta_x_hat_single = jax.vmap(
                    lambda z, u: self.tf_policy.apply(
                        tf_params_dyn_sg, z, u,
                        method=lambda mdl, zz, uu: mdl.call_dynamics_head(zz, uu),
                    )
                )(z_t_batch, u_clipped)                                  # [mb, n_agents, state_dim]
                # clip_state → reproduce the real env (x_next = clip(x + increment))
                # and stay consistent with the horizon-rollout step-0 state.
                # dyn_scale de-normalises the head output to physical units
                # (no-op when dyn_scale == 1, i.e. DI / DubinsCar).
                x_hat_1 = self._env.clip_state(
                    x_t + delta_x_hat_single * self._dyn_scale[None, None, :]
                )                                                        # [mb, n_agents, state_dim]

                def obs_next_fn(x_next_i, goal_i, obstacle_i):
                    return self._env._get_obs(x_next_i, goal_i, obstacle_i)

                # Signed-distance encoding: algebraic SDF, no zero-norm NaN → full grad.
                obs_next = jax.vmap(obs_next_fn)(x_hat_1, goal_batch, obstacle_batch)
                # obs_next: [mb, n_agents, obs_dim]

                h_next_full    = jax.vmap(eval_cbf)(obs_next)         # [mb, n_agents] — full grad
                h_next_stopped = jax.vmap(eval_cbf_stopped)(obs_next) # [mb, n_agents] — CBF grad stopped
                # Use full grad on labeled samples (train CBF + actor), stopped on unlabeled (train actor only)
                h_next = jnp.where(labeled_mask, h_next_full, h_next_stopped)

                c_single    = self.gamma - h_next + (1.0 - self.alpha) * h  # [mb, n_agents]
                loss_dt_cbf = jnp.mean(jax.nn.relu(c_single + self.eps))
                acc_dt_cbf  = jnp.mean(c_single <= 0.0)


                u_lb_s, u_ub_s = self._env.action_lim()
                tf_horizon = self.horizon   # H (static int, safe inside JIT)


                def _tf_fwd_roll(obs_s, act_s, u_n):
                    # Single-step transformer fwd: returns the applied action AND the
                    # dynamics-head increment, so the rollout state is advanced by the
                    # dynamics head exactly as at deployment (paper Eq. 10).
                    # tf_params_dyn_sg: stop_gradient on dyn-head params only, so the
                    # horizon CBF loss trains θ_T and θ_u (via z_t and u_applied) but
                    # NOT θ_x — dyn head stays a frozen sim surrogate here.
                    out = self.tf_policy.apply(
                        tf_params_dyn_sg, obs_s, act_s, u_n, True,
                        rngs={'dropout': step_key},
                    )
                    return out.u_applied, out.delta_x_hat

                def scan_step(carry, _):
                    x_k, obs_k, act_k = carry

                    u_nom_k = jax.vmap(self._env.u_nom_from_state)(x_k, goal_batch)
                    u_app_k, dx_hat_k = jax.vmap(_tf_fwd_roll)(obs_k, act_k, u_nom_k)

                    u_clip_k = jnp.clip(u_app_k, u_lb_s[None, None, :], u_ub_s[None, None, :])

                    # ── Use DYNAMICS HEAD to advance state — not the simulator. ──
                    # Matches deployment (algo/rollout.autoregressive_rollout) and the
                    # paper's Eq. (10): x̂_{k+1} = clip( x̂_k + d_θx(z_k, u_k) ).
                    # The dyn head predicts the UNCLIPPED Euler increment (its
                    # loss_dyn target is step_simulator = x + xdot*dt), so the real
                    # env's composition  x_next = clip_state(x + xdot*dt)  is only
                    # reproduced if clip_state is applied here too.  Without it the
                    # rollout drifts to velocities |v| > v_max the environment can
                    # never reach, and both the CBF eval at each step and the dyn
                    # head input at the next step are fed unreachable states.
                    x_k = self._env.clip_state(
                        x_k + dx_hat_k * self._dyn_scale[None, None, :]
                    )

                    o_hat_k = jax.vmap(obs_next_fn)(x_k, goal_batch, obstacle_batch)
                    obs_k = jnp.concatenate([obs_k[:, :, 1:, :], o_hat_k[:, :, None, :]], axis=2)
                    act_k = jnp.concatenate([act_k[:, :, 1:, :], u_clip_k[:, :, None, :]], axis=2)
                    return (x_k, obs_k, act_k), x_k

                _, x_hat_steps = lax.scan(
                    scan_step,
                    (x_t, obs_seq, action_seq),
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

                    h_seq_full = jax.vmap(h_at)(x_h_i)          # [H+1, n_agents]
                    h_seq_stop = jax.vmap(h_at_stop_cbf)(x_h_i) # [H+1, n_agents]

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
                    v_k = jax.nn.relu(c_k + self.eps)          # [H, n_agents]
                    H_len = v_k.shape[0]
                    return (
                        jax.scipy.special.logsumexp(self.beta * v_k + 1e-8, axis=0)
                        / self.beta
                        - jnp.log(H_len) / self.beta
                    )                                           # [n_agents]

                horizon_viol = jax.vmap(eval_horizon_sample)(
                    x_hat_horizon_tf,  # [mb, H+1, n_agents, 4]  ← transformer rollout
                    goal_batch,
                    obstacle_batch,
                    labeled_mask,
                )                                               # [mb, n_agents]
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

            # Frozen-backbone ablation: zero the non-actor gradient subtrees
            # BEFORE clipping.  optax.set_to_zero already blocks the update, but
            # leaving the (soon-to-be-zeroed) loss_dyn gradient on theta_T /
            # theta_x in place would inflate the global norm and throttle the
            # actor-head step, and a non-finite frozen-tree gradient could make
            # apply_if_finite skip the actor update entirely.  No-op when the
            # flag is off.
            if self.freeze_backbone:
                grad_tf = {'params': {
                    k: (v if k == 'action_head' else jtu.tree_map(jnp.zeros_like, v))
                    for k, v in grad_tf['params'].items()
                }}

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
        # Prefix mean keys with "mean/" so both appear in wandb without collision.
        info_combined = {
            **{f"mean/{k}": v for k, v in info_mean.items()},
            **info_last,
        }
        return cbf_mlp_train_state, tf_actor_train_state, info_combined

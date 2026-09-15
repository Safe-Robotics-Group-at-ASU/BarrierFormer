import jax.tree_util as jtu
import numpy as np

from abc import ABC, abstractproperty, abstractmethod
from .data import Rollout
from .utils import jax2np, np2jax
from ..utils.utils import tree_merge
from ..utils.typing import Array


class Buffer(ABC):

    def __init__(self, size: int):
        self._size = size

    @abstractmethod
    def append(self, rollout: Rollout):
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Rollout:
        pass

    @abstractproperty
    def length(self) -> int:
        pass


class ReplayBuffer(Buffer):

    def __init__(self, size: int):
        super(ReplayBuffer, self).__init__(size)
        self._buffer = None

    def append(self, rollout: Rollout):
        if self._buffer is None:
            self._buffer = jax2np(rollout)
        else:
            self._buffer = tree_merge([self._buffer, jax2np(rollout)])
        if self._buffer.length > self._size:
            self._buffer = jtu.tree_map(lambda x: x[-self._size:], self._buffer)

    def sample(self, batch_size: int) -> Rollout:
        idx = np.random.randint(0, self._buffer.length, batch_size)
        return np2jax(self.get_data(idx))

    def get_data(self, idx: np.ndarray) -> Rollout:
        return jtu.tree_map(lambda x: x[idx], self._buffer)

    @property
    def length(self) -> int:
        if self._buffer is None:
            return 0
        return self._buffer.n_data


class MaskedReplayBuffer:
    """
    Replay buffer that stores rollout episodes together with **two**
    per-transition safety masks (safe / unsafe).

    Storage layout
    --------------
    _buffer          : Rollout  shape (n_episodes, T, ...)
    _safe_mask       : ndarray  shape (n_episodes, T, n_agents)   — D_S
    _unsafe_mask     : ndarray  shape (n_episodes, T, n_agents)   — D_U

    Paper **D_B+** (non–known-unsafe) is always ``~unsafe_mask`` (safe ∪
    unlabeled).  It is **not** stored — derive if you need it downstream.

    Training in ``BarrierFormer`` recomputes ``labeled_mask = safe ∨ unsafe`` in
    ``update_inner_transformer`` for CBF future-gradient gating (legacy GNN
    ``h_dot`` parity); it does not rely on a stored D_B+ tensor.

    _size is measured in EPISODES (not transitions).

    Sampling
    --------
    Each call to sample() draws independent (episode_idx, timestep_idx) pairs
    and returns individual flat transitions of shape (batch_size, ...).

    This is correct for the transformer because every graph already carries
    its full obs_history / action_history context window inside env_states —
    no episode-aware window extraction is needed.
    """

    def __init__(self, size: int):
        self._size = size          # maximum number of stored episodes
        self._buffer = None
        self._safe_mask = None
        self._unsafe_mask = None

    # ------------------------------------------------------------------
    def append(
        self,
        rollout: Rollout,
        safe_mask: Array,
        unsafe_mask: Array,
    ):
        """
        Add a batch of episodes to the buffer.

        Parameters
        ----------
        rollout         : Rollout  (n_env, T, ...)
        safe_mask       : Array    (n_env, T, n_agents)  bool  D_S labels
        unsafe_mask     : Array    (n_env, T, n_agents)  bool  D_U labels
        """
        rollout_np        = jax2np(rollout)
        safe_np           = np.asarray(jax2np(safe_mask))
        unsafe_np         = np.asarray(jax2np(unsafe_mask))

        if self._buffer is None:
            self._buffer      = rollout_np
            self._safe_mask   = safe_np
            self._unsafe_mask = unsafe_np
        else:
            self._buffer      = tree_merge([self._buffer, rollout_np])
            self._safe_mask   = np.concatenate([self._safe_mask,   safe_np],   axis=0)
            self._unsafe_mask = np.concatenate([self._unsafe_mask, unsafe_np], axis=0)

        # Evict the oldest episodes when over capacity.
        if self._buffer.length > self._size:
            self._buffer      = jtu.tree_map(lambda x: x[-self._size:], self._buffer)
            self._safe_mask   = self._safe_mask[-self._size:]
            self._unsafe_mask = self._unsafe_mask[-self._size:]

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> tuple[Rollout, Array, Array]:
        """
        Sample batch_size individual transitions.

        Draws random (episode_idx, timestep_idx) pairs so that every
        stored transition has equal probability of being selected,
        then returns the corresponding flat batch.

        Returns
        -------
        (rollout, safe_mask, unsafe_mask)
        each with leading dimension batch_size, as JAX arrays.
        """
        n_eps = self._buffer.length         # number of stored episodes
        T     = self._buffer.time_horizon   # steps per episode

        ep_idx = np.random.randint(0, n_eps, batch_size)
        t_idx  = np.random.randint(0, T,     batch_size)

        return self.get_data(ep_idx, t_idx)

    def get_data(
        self, ep_idx: np.ndarray, t_idx: np.ndarray
    ) -> tuple[Rollout, Array, Array]:
        """
        Extract transitions at (ep_idx[i], t_idx[i]) for each i.

        x[ep_idx, t_idx] selects one element per (episode, timestep)
        pair from arrays of shape (n_episodes, T, ...), giving a flat
        batch of shape (batch_size, ...).
        """
        rollout     = np2jax(jtu.tree_map(lambda x: x[ep_idx, t_idx], self._buffer))
        safe_mask   = np2jax(self._safe_mask[ep_idx, t_idx])
        unsafe_mask = np2jax(self._unsafe_mask[ep_idx, t_idx])
        return rollout, safe_mask, unsafe_mask

    # ------------------------------------------------------------------
    @property
    def length(self) -> int:
        """
        Total number of flat transitions available for sampling
        (n_episodes × T).  Used for the 'enough data?' guard.
        """
        if self._buffer is None:
            return 0
        return self._buffer.n_data

    @property
    def n_episodes(self) -> int:
        """Number of complete episodes currently stored."""
        if self._buffer is None:
            return 0
        return self._buffer.length

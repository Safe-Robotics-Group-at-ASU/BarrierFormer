import jax.tree_util as jtu
import numpy as np

from abc import ABC, abstractproperty, abstractmethod
from .data import Rollout
from .utils import jax2np, np2jax
from ..utils.utils import tree_merge
from ..utils.typing import Array


class Buffer(ABC):

    """Abstract replay buffer."""
    def __init__(self, size: int):
        """Set the buffer size."""
        self._size = size

    @abstractmethod
    def append(self, rollout: Rollout):
        """Add data to the buffer."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Rollout:
        """Sample a batch."""
        pass

    @abstractproperty
    def length(self) -> int:
        """Number of stored items."""
        pass


class ReplayBuffer(Buffer):

    """FIFO replay buffer of rollouts."""
    def __init__(self, size: int):
        """Create an empty buffer."""
        super(ReplayBuffer, self).__init__(size)
        self._buffer = None

    def append(self, rollout: Rollout):
        """Add rollouts, dropping the oldest when full."""
        if self._buffer is None:
            self._buffer = jax2np(rollout)
        else:
            self._buffer = tree_merge([self._buffer, jax2np(rollout)])
        if self._buffer.length > self._size:
            self._buffer = jtu.tree_map(lambda x: x[-self._size:], self._buffer)

    def sample(self, batch_size: int) -> Rollout:
        """Sample random rollouts."""
        idx = np.random.randint(0, self._buffer.length, batch_size)
        return np2jax(self.get_data(idx))

    def get_data(self, idx: np.ndarray) -> Rollout:
        """Get rollouts by index."""
        return jtu.tree_map(lambda x: x[idx], self._buffer)

    @property
    def length(self) -> int:
        """Number of stored rollouts."""
        if self._buffer is None:
            return 0
        return self._buffer.n_data


class MaskedReplayBuffer:
    """Replay buffer that stores rollout episodes together with **two** per-transition safety masks (safe / unsafe)."""

    def __init__(self, size: int):
        """Create an empty buffer."""
        self._size = size
        self._buffer = None
        self._safe_mask = None
        self._unsafe_mask = None

    def append(
        self,
        rollout: Rollout,
        safe_mask: Array,
        unsafe_mask: Array,
    ):
        """Add a batch of episodes to the buffer."""
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

        if self._buffer.length > self._size:
            self._buffer      = jtu.tree_map(lambda x: x[-self._size:], self._buffer)
            self._safe_mask   = self._safe_mask[-self._size:]
            self._unsafe_mask = self._unsafe_mask[-self._size:]

    def sample(self, batch_size: int) -> tuple[Rollout, Array, Array]:
        """Sample batch_size individual transitions."""
        n_eps = self._buffer.length
        T     = self._buffer.time_horizon

        ep_idx = np.random.randint(0, n_eps, batch_size)
        t_idx  = np.random.randint(0, T,     batch_size)

        return self.get_data(ep_idx, t_idx)

    def get_data(
        self, ep_idx: np.ndarray, t_idx: np.ndarray
    ) -> tuple[Rollout, Array, Array]:
        """Extract transitions at (ep_idx[i], t_idx[i]) for each i."""
        rollout     = np2jax(jtu.tree_map(lambda x: x[ep_idx, t_idx], self._buffer))
        safe_mask   = np2jax(self._safe_mask[ep_idx, t_idx])
        unsafe_mask = np2jax(self._unsafe_mask[ep_idx, t_idx])
        return rollout, safe_mask, unsafe_mask

    @property
    def length(self) -> int:
        """Total number of flat transitions available for sampling (n_episodes × T)."""
        if self._buffer is None:
            return 0
        return self._buffer.n_data

    @property
    def n_episodes(self) -> int:
        """Number of complete episodes currently stored."""
        if self._buffer is None:
            return 0
        return self._buffer.length


class MaskedTransitionBuffer:
    """Flat store of individual transitions with their safe / unsafe masks."""

    def __init__(self, size: int):
        """Create an empty buffer."""
        self._size = size
        self._buffer = None
        self._safe_mask = None
        self._unsafe_mask = None

    def append(self, transitions: Rollout, safe_mask: Array, unsafe_mask: Array):
        """Add flat transitions: every leaf (n, ...), masks (n, n_agents)."""
        transitions = jax2np(transitions)
        safe_np     = np.asarray(jax2np(safe_mask))
        unsafe_np   = np.asarray(jax2np(unsafe_mask))
        if self._buffer is None:
            self._buffer, self._safe_mask, self._unsafe_mask = transitions, safe_np, unsafe_np
        else:
            self._buffer      = tree_merge([self._buffer, transitions])
            self._safe_mask   = np.concatenate([self._safe_mask,   safe_np],   axis=0)
            self._unsafe_mask = np.concatenate([self._unsafe_mask, unsafe_np], axis=0)
        if self.length > self._size:
            self._buffer      = jtu.tree_map(lambda x: x[-self._size:], self._buffer)
            self._safe_mask   = self._safe_mask[-self._size:]
            self._unsafe_mask = self._unsafe_mask[-self._size:]

    def sample(self, batch_size: int) -> tuple[Rollout, Array, Array]:
        """batch_size transitions drawn uniformly WITH replacement."""
        idx = np.random.randint(0, self.length, batch_size)
        return (np2jax(jtu.tree_map(lambda x: x[idx], self._buffer)),
                np2jax(self._safe_mask[idx]), np2jax(self._unsafe_mask[idx]))

    @property
    def length(self) -> int:
        """Number of stored transitions."""
        if self._buffer is None:
            return 0
        return int(self._safe_mask.shape[0])

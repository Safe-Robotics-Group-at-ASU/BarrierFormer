from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, Tuple

from barrierformer.utils.typing import Action, Params, PRNGKey, Array
from barrierformer.utils.graph import GraphsTuple
from barrierformer.trainer.data import Rollout
from barrierformer.env.base import MultiAgentEnv


class MultiAgentController(ABC):

    """Abstract base class for multi-agent controllers."""
    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            action_dim: int,
            n_agents: int
    ):
        """Store the environment and graph/action dimensions."""
        self._env = env
        self._node_dim = node_dim
        self._edge_dim = edge_dim
        self._action_dim = action_dim
        self._n_agents = n_agents

    @property
    def node_dim(self) -> int:
        """Node feature dimension."""
        return self._node_dim

    @property
    def edge_dim(self) -> int:
        """Edge feature dimension."""
        return self._edge_dim

    @property
    def action_dim(self) -> int:
        """Action dimension."""
        return self._action_dim

    @property
    def n_agents(self) -> int:
        """Number of agents."""
        return self._n_agents

    @abstractproperty
    def config(self) -> dict:
        """Algorithm configuration dictionary."""
        pass

    @abstractproperty
    def actor_params(self) -> Params:
        """Current actor parameters."""
        pass

    @abstractmethod
    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        """Deterministic action for a graph."""
        pass

    @abstractmethod
    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        """Stochastic action for a graph (used during data collection)."""
        pass

    @abstractmethod
    def update(self, rollout: Rollout, step: int) -> dict:
        """Update the networks from a rollout."""
        pass

    @abstractmethod
    def save(self, save_dir: str, step: int):
        """Save the model parameters."""
        pass

    @abstractmethod
    def load(self, load_dir: str, step: int):
        """Load the model parameters."""
        pass

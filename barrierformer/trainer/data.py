from typing import NamedTuple

from ..utils.typing import Array
from ..utils.typing import Action, Reward, Cost, Done
from ..utils.graph import GraphsTuple


class Rollout(NamedTuple):
    """Batch of episode trajectories."""
    graph: GraphsTuple
    actions: Action
    rewards: Reward
    costs: Cost
    dones: Done
    next_graph: GraphsTuple

    @property
    def length(self) -> int:
        """Number of episodes."""
        return self.rewards.shape[0]

    @property
    def time_horizon(self) -> int:
        """Number of time steps per episode."""
        return self.rewards.shape[1]

    @property
    def num_agents(self) -> int:
        """Number of agents."""
        return self.rewards.shape[2]

    @property
    def n_data(self) -> int:
        """Total number of transitions."""
        return self.length * self.time_horizon

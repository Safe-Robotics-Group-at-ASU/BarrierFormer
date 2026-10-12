import functools as ft
import numpy as np
import pathlib
import jax
import jax.lax as lax
import jax.numpy as jnp
import tqdm

from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, NamedTuple, Optional, Tuple

from ..utils.graph import GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, PRNGKey, Reward, State
from ..utils.utils import jax2np, jax_jit_np, tree_concat_at_front, tree_stack


class StepResult(NamedTuple):
    """Result of one environment step."""
    graph: GraphsTuple
    reward: Reward
    cost: Cost
    done: Done
    info: Info


class RolloutResult(NamedTuple):
    """Result of a full episode rollout."""
    Tp1_graph: GraphsTuple
    T_action: Action
    T_reward: Reward
    T_cost: Cost
    T_done: Done
    T_info: Info


class MultiAgentEnv(ABC):

    """Abstract base class for multi-agent graph environments."""
    PARAMS = {}

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.03,
            params: dict = None
    ):
        """Store the environment settings."""
        super(MultiAgentEnv, self).__init__()
        self._num_agents = num_agents
        self._dt = dt
        if params is None:
            params = self.PARAMS
        self._params = params
        self._t = 0
        self._max_step = max_step
        self._max_travel = max_travel
        self._area_size = area_size

    @property
    def params(self) -> dict:
        """Environment parameters."""
        return self._params

    @property
    def num_agents(self) -> int:
        """Number of agents."""
        return self._num_agents

    @property
    def max_travel(self) -> float:
        """Maximum travel distance."""
        return self._max_travel

    @property
    def area_size(self) -> float:
        """Side length of the workspace."""
        return self._area_size

    @property
    def dt(self) -> float:
        """Simulation time step."""
        return self._dt

    @property
    def max_episode_steps(self) -> int:
        """Episode length."""
        return self._max_step

    def clip_state(self, state: State) -> State:
        """Clip states to the state limits."""
        lower_limit, upper_limit = self.state_lim(state)
        return jnp.clip(state, lower_limit, upper_limit)

    def clip_action(self, action: Action) -> Action:
        """Clip actions to the action limits."""
        lower_limit, upper_limit = self.action_lim()
        return jnp.clip(action, lower_limit, upper_limit)

    @abstractproperty
    def state_dim(self) -> int:
        """Dimension of one agent's state."""
        pass

    @abstractproperty
    def node_dim(self) -> int:
        """Dimension of a graph node feature."""
        pass

    @abstractproperty
    def edge_dim(self) -> int:
        """Dimension of a graph edge feature."""
        pass

    @abstractproperty
    def action_dim(self) -> int:
        """Dimension of one agent's action."""
        pass

    @abstractmethod
    def reset(self, key: Array) -> GraphsTuple:
        """Sample a new episode (obstacles, start and goal) and return its graph."""
        pass

    def reset_np(self, key: Array) -> GraphsTuple:
        """Reset, but without the constraint that it has to be jittable."""
        return self.reset(key)

    @abstractmethod
    def step(self, graph: GraphsTuple, action: Action, get_eval_info: bool = False) -> StepResult:
        """Advance the environment one step and return the next graph, reward, cost and done flag."""
        pass

    @abstractmethod
    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """Lower and upper limits of the agent state."""
        pass

    @abstractmethod
    def action_lim(self) -> Tuple[Action, Action]:
        """Lower and upper limits of the agent action."""
        pass

    @abstractmethod
    def control_affine_dyn(self, state: State) -> [Array, Array]:
        """Control-affine dynamics terms f(x) and g(x)."""
        pass

    @abstractmethod
    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        """Add edge features to a graph from the given states."""
        pass

    @abstractmethod
    def get_graph(self, state: State) -> GraphsTuple:
        """Build the graph for the given environment state."""
        pass

    @abstractmethod
    def u_ref(self, graph: GraphsTuple) -> Action:
        """Nominal (reference) action for each agent."""
        pass

    @abstractmethod
    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        """Graph obtained by applying the action to the current graph."""
        pass

    @abstractmethod
    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, graph: GraphsTuple) -> Array:
        """Mask of agents that are clearly safe."""
        pass

    @abstractmethod
    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        """Mask of agents that are in collision or about to collide."""
        pass

    @abstractmethod
    def collision_mask(self, graph: GraphsTuple) -> Array:
        """Mask of agents currently colliding with an obstacle."""
        pass

    def rollout_fn(self, policy: Callable, rollout_length: int = None) -> Callable[[PRNGKey], RolloutResult]:
        """Build a function that rolls out one episode with a policy."""
        rollout_length = rollout_length or self.max_episode_steps

        def body(graph, _):
            action = policy(graph)
            graph_new, reward, cost, done, info = self.step(graph, action, get_eval_info=True)
            return graph_new, (graph_new, action, reward, cost, done, info)

        def fn(key: PRNGKey) -> RolloutResult:
            graph0 = self.reset(key)
            graph_final, (T_graph, T_action, T_reward, T_cost, T_done, T_info) = lax.scan(
                body, graph0, None, length=rollout_length
            )
            Tp1_graph = tree_concat_at_front(graph0, T_graph, axis=0)

            return RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info)

        return fn

    def rollout_fn_jitstep(
        self, policy: Callable, rollout_length: int = None, noedge: bool = False, nograph: bool = False
    ):
        """Build a rollout function with a jitted step (python loop)."""
        rollout_length = rollout_length or self.max_episode_steps

        def body(graph, _):
            action = policy(graph)
            graph_new, reward, cost, done, info = self.step(graph, action, get_eval_info=True)
            return graph_new, (graph_new, action, reward, cost, done, info)

        jit_body = jax.jit(body)

        is_unsafe_fn = jax_jit_np(self.collision_mask)
        is_finish_fn = jax_jit_np(self.finish_mask)

        def fn(key: PRNGKey) -> [RolloutResult, Array, Array]:
            graph0 = self.reset_np(key)
            graph = graph0
            T_output = []
            is_unsafes = []
            is_finishes = []

            is_unsafes.append(is_unsafe_fn(graph0))
            is_finishes.append(is_finish_fn(graph0))
            graph0 = jax2np(graph0)

            for kk in tqdm.trange(rollout_length, ncols=80):
                graph, output = jit_body(graph, None)

                is_unsafes.append(is_unsafe_fn(graph))
                is_finishes.append(is_finish_fn(graph))

                output = jax2np(output)
                if noedge:
                    output = (output[0].without_edge(), *output[1:])
                if nograph:
                    output = (None, *output[1:])
                T_output.append(output)

            T_graph = [o[0] for o in T_output]
            if noedge:
                T_graph = [graph0.without_edge()] + T_graph
            else:
                T_graph = [graph0] + T_graph
            del graph0
            T_action = [o[1] for o in T_output]
            T_reward = [o[2] for o in T_output]
            T_cost = [o[3] for o in T_output]
            T_done = [o[4] for o in T_output]
            T_info = [o[5] for o in T_output]
            del T_output

            if nograph:
                T_graph = None
            else:
                T_graph = tree_stack(T_graph)
            T_action = tree_stack(T_action)
            T_reward = tree_stack(T_reward)
            T_cost = tree_stack(T_cost)
            T_done = tree_stack(T_done)
            T_info = tree_stack(T_info)

            Tp1_graph = T_graph

            rollout_result = jax2np(RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info))
            return rollout_result, np.stack(is_unsafes, axis=0), np.stack(is_finishes, axis=0)

        return fn

    @abstractmethod
    def render_video(
        self, rollout: RolloutResult, video_path: pathlib.Path, Ta_is_unsafe=None, viz_opts: dict = None, **kwargs
    ) -> None:
        """Render an episode rollout to a video file."""
        pass

    @abstractmethod
    def finish_mask(self, graph: GraphsTuple) -> Array:
        """Mask of agents that have reached their goal."""
        pass

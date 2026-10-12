import functools as ft
import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from typing import Literal, NamedTuple, Tuple, Optional

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, AgentState, Array, Cost, Done, Info, Reward, State
from ..utils.utils import merge01, jax_vmap
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
from .plot import render_video
from .utils import get_lidar, inside_obstacles, lqr, get_node_goal_rng

def build_transformer_tokens(
    obs_history: Array,
    action_history: Array,
    mode: Literal['zero_pad', 'nominal'],
    u_nom: Array = None,
) -> Array:
    """Construct the per-timestep token sequence fed to the causal transformer."""
    if mode not in ('zero_pad', 'nominal'):
        raise ValueError(f"mode must be 'zero_pad' or 'nominal', got '{mode}'")
    if mode == 'nominal' and u_nom is None:
        raise ValueError("u_nom must be provided when mode='nominal'")

    T        = obs_history.shape[0]
    n_agents = obs_history.shape[1]
    act_dim  = action_history.shape[2]

    obs_seq = obs_history.transpose(1, 0, 2)

    act_seq = action_history.transpose(1, 0, 2)

    if mode == 'zero_pad':
        current_action = jnp.zeros((n_agents, 1, act_dim))
    else:
        current_action = u_nom[:, None, :]

    act_padded = jnp.concatenate([act_seq, current_action], axis=1)

    return jnp.concatenate([obs_seq, act_padded], axis=-1)


class DoubleIntegrator(MultiAgentEnv):
    """2D double-integrator environment with rectangular obstacles and LiDAR."""
    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        """Environment state: agent states, goals and obstacles."""
        agent: AgentState
        goal: State
        obstacle: Obstacle
        obs_history: Array
        action_history: Array
        step_count: Array

        @property
        def n_agent(self) -> int:
            """Number of agents."""
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.5],
        "n_obs": 8,
        "m": 0.1,
        "safe_buffer_mult": 1.0,
        "lidar_ray_order": False,
    }

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.03,
            params: dict = None,
            history_len: int = 12
    ):
        """Create the environment with the given parameters."""
        super(DoubleIntegrator, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self.history_len = history_len
        A = np.zeros((self.state_dim, self.state_dim), dtype=np.float32)
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        self._A = A * self._dt + np.eye(self.state_dim)
        self._B = (
            np.array([[0.0, 0.0], [0.0, 0.0], [1.0 / self._params["m"], 0.0], [0.0, 1.0 / self._params["m"]]])
            * self._dt
        )
        self._Q = np.eye(self.state_dim) * 5
        self._R = np.eye(self.action_dim)
        self._K = jnp.array(lqr(self._A, self._B, self._Q, self._R))
        self.create_obstacles = jax_vmap(Rectangle.create)
        n_rays = self._params["n_rays"]
        _thetas = np.linspace(-np.pi, np.pi - 2 * np.pi / n_rays, n_rays)
        self._ray_cos = jnp.array(np.cos(_thetas))
        self._ray_sin = jnp.array(np.sin(_thetas))

    @property
    def state_dim(self) -> int:
        """Dimension of one agent's state."""
        return 4

    @property
    def node_dim(self) -> int:
        """Dimension of a graph node feature."""
        return 3

    @property
    def edge_dim(self) -> int:
        """Dimension of a graph edge feature."""
        return 4
    
    @property
    def action_dim(self) -> int:
        """Dimension of one agent's action."""
        return 2

    @property
    def obs_dim(self) -> int:
        """Dimension of the per-agent observation vector."""
        return self.state_dim + 2 + self._params["n_rays"] * 4

    @property
    def token_dim(self) -> int:
        """Dimension of one transformer input token."""
        return self.obs_dim + self.action_dim

    def reset(self, key: Array) -> GraphsTuple:
        """Sample a new episode (obstacles, start and goal) and return its graph."""
        self._t = 0

        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        min_obs_sep = self._params["obs_len_range"][1] * 2.0

        obstacle_key, key = jr.split(key, 2)

        sentinel = jnp.ones((n_rng_obs, 2)) * (self.area_size * 100.0)

        def place_one(carry, idx):
            """Place one obstacle center, rejection-sampling against already-placed centers."""
            placed, step_key = carry
            obs_key, step_key = jr.split(step_key)
            init_cand = jr.uniform(obs_key, (2,), minval=0.0, maxval=self.area_size)

            def too_close(state):
                i, _k, cand = state
                dists = jnp.linalg.norm(placed - cand, axis=1)
                return (dists < min_obs_sep).any() & (i < 200)

            def resample(state):
                i, k, _cand = state
                use_key, k = jr.split(k)
                new_cand = jr.uniform(use_key, (2,), minval=0.0, maxval=self.area_size)
                return i + 1, k, new_cand

            _, _, final_pos = jax.lax.while_loop(too_close, resample, (0, obs_key, init_cand))
            new_placed = placed.at[idx].set(final_pos)
            return (new_placed, step_key), final_pos

        (_, obstacle_key), obs_pos = jax.lax.scan(
            place_one, (sentinel, obstacle_key), jnp.arange(n_rng_obs)
        )

        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (n_rng_obs, 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
        obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        states, goals = get_node_goal_rng(
            key, self.area_size, 2, obstacles, self.num_agents, 4 * self.params["car_radius"], self.max_travel)

        states = jnp.concatenate([states, jnp.zeros((self.num_agents, 2))], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, 2))], axis=1)

        initial_obs = self._get_obs(states, goals, obstacles)
        obs_history = jnp.repeat(initial_obs[None, ...], self.history_len, axis=0)
        action_history = jnp.zeros((self.history_len - 1, self.num_agents, self.action_dim))

        env_states = self.EnvState(
            agent=states,
            goal=goals,
            obstacle=obstacles,
            obs_history=obs_history,
            action_history=action_history,
            step_count=jnp.array(0, dtype=jnp.int32)
        )

        return self.get_graph(env_states)

    def agent_accel(self, action: Action) -> Action:
        """Acceleration from force."""
        return action / self._params["m"]

    def agent_step_exact(self, agent_states: AgentState, action: Action) -> AgentState:
        """Exact discrete-time double-integrator step."""
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        n_accel = self.agent_accel(action)
        n_pos_new = agent_states[:, :2] + agent_states[:, 2:] * self.dt + n_accel * self.dt**2 / 2
        n_vel_new = agent_states[:, 2:] + n_accel * self.dt
        n_state_agent_new = jnp.concatenate([n_pos_new, n_vel_new], axis=1)
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return n_state_agent_new

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        """One Euler integration step of the agent dynamics."""
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = self.agent_xdot(agent_states, action)
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def step_simulator(self, agent_states: AgentState, action: Action) -> AgentState:
        """Pure JAX dynamics step for SQP / predictive-rollout use."""
        x_dot = self.agent_xdot(agent_states, action)
        return agent_states + x_dot * self.dt

    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        """Continuous-time agent dynamics x_dot = f(x, u)."""
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        n_accel = self.agent_accel(action)
        x_dot = jnp.concatenate([agent_states[:, 2:], n_accel], axis=1)
        assert x_dot.shape == (self.num_agents, self.state_dim)
        return x_dot

    def step(
        self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        """Advance the environment one step and return the next graph, reward, cost and done flag."""
        self._t += 1

        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)
        next_obs = self._get_obs(next_agent_states, goal_states, obstacles)

        obs_history = graph.env_states.obs_history
        action_history = graph.env_states.action_history
        step_count = graph.env_states.step_count

        new_obs_history = jnp.concatenate([obs_history[1:], next_obs[None, ...]], axis=0)
        new_action_history = jnp.concatenate([action_history[1:], action[None, ...]], axis=0)

        done = jnp.array(False)

        reward = jnp.zeros(()).astype(jnp.float32)
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        cost = self.get_cost(graph)
        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()
        next_state = self.EnvState(
            agent=next_agent_states,
            goal=goal_states,
            obstacle=obstacles,
            obs_history=new_obs_history,
            action_history=new_action_history,
            step_count=step_count + 1
        )

        info = {}
        if get_eval_info:
            agent_pos = agent_states[:, :2]
            info["inside_obstacles"] = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])
            info["transformer_obs_history"] = new_obs_history
            info["transformer_act_history"] = new_action_history
        return self.get_graph(next_state), reward, cost, done, info

    def get_cost(self, graph: EnvGraphsTuple) -> Cost:
        """Collision cost of the current graph."""
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle

        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        collision = (self._params["car_radius"] * 2 > dist).any(axis=1)
        cost = collision.mean()

        collision = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])
        cost += collision.mean()

        return cost

    def render_video(
            self,
            rollout: RolloutResult,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        """Render an episode rollout to a video file."""
        render_video(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            n_rays=self.params["n_rays"],
            r=self.params["car_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            **kwargs
        )

    def edge_blocks(self, state: EnvState, lidar_data: State) -> list[EdgeBlock]:
        """Build the agent-agent, agent-goal and agent-obstacle edge blocks."""
        n_hits = self._params["n_rays"] * self.num_agents

        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        state_diff = state.agent[:, None, :] - state.agent[None, :, :]
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.eye(self.num_agents)
        agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :2].set(agent_goal_feats[:, :2] * coef)
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self._params["n_rays"], (i + 1) * self._params["n_rays"])
            lidar_pos = agent_pos[i, :] - lidar_data[id_hits, :2]
            lidar_feats = state.agent[i, :] - lidar_data[id_hits, :]
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            agent_obs_mask = jnp.ones((1, self._params["n_rays"]))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        """Control-affine dynamics terms f(x) and g(x)."""
        assert state.ndim == 2
        f = jnp.concatenate([state[:, 2:], jnp.zeros((state.shape[0], 2))], axis=1)
        g = jnp.concatenate([jnp.zeros((2, 2)), jnp.eye(2) / self._params['m']], axis=0)
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], self.state_dim, self.action_dim)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        """Add edge features to a graph from the given states."""
        assert graph.is_single
        assert state.ndim == 2

        edge_feats = state[graph.receivers] - state[graph.senders]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :2].set(edge_feats[:, :2] * coef)

        return graph._replace(edges=edge_feats, states=state)

    def _get_obs(self, agent_states: AgentState, goal_states: State, obstacles) -> Array:
        """Build the per-agent observation [state | goal offset | LiDAR features]."""
        sense_range = self._params["comm_radius"]
        get_lidar_fn = jax_vmap(
            ft.partial(
                get_lidar,
                obstacles=obstacles,
                num_beams=self._params["n_rays"],
                sense_range=sense_range,
                max_returns=self._params["n_rays"],
                sort=not self._params.get("lidar_ray_order", False),
            )
        )
        lidar_xy = get_lidar_fn(agent_states[:, :2])

        def encode_agent(agent_pos, lidar_hits):
            hits_sg   = jax.lax.stop_gradient(lidar_hits)
            rel       = hits_sg - agent_pos[None, :]
            dist      = jnp.sqrt(jnp.sum(rel ** 2, axis=-1) + 1e-8)
            hit_mask  = jnp.less(dist, sense_range).astype(jnp.float32)

            all_sdf  = jax.vmap(lambda obs_j: obs_j.signed_distance(agent_pos))(obstacles)
            min_sdf  = all_sdf.min()
            is_inside = min_sdf < 0.0

            dist_norm = jnp.where(
                is_inside,
                jnp.clip(min_sdf / sense_range, -1.0, 0.0),
                jnp.clip(jnp.where(hit_mask > 0.5, dist / sense_range, 1.0), 0.0, 1.0),
            )
            encoded   = jnp.stack(
                [hit_mask, dist_norm, self._ray_cos, self._ray_sin], axis=-1)
            return encoded.reshape(-1)

        lidar_encoded = jax.vmap(encode_agent)(
            agent_states[:, :2], lidar_xy)
        rel_goal = goal_states[:, :2] - agent_states[:, :2]
        return jnp.concatenate(
            [agent_states, rel_goal, lidar_encoded], axis=-1)

    def get_transformer_tokens(
        self,
        graph: 'DoubleIntegrator.EnvGraphsTuple',
        mode: Literal['zero_pad', 'nominal'] = 'nominal',
    ) -> Array:
        """Build the transformer input token sequence from the current graph."""
        obs_history    = graph.env_states.obs_history
        action_history = graph.env_states.action_history

        u_nom = self.u_ref(graph) if mode == 'nominal' else None

        return build_transformer_tokens(obs_history, action_history, mode, u_nom)

    def get_graph(self, state: EnvState, adjacency: Array = None) -> GraphsTuple:
        """Build the graph for the given environment state."""
        n_hits = self._params["n_rays"] * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, 3))
        node_feats = node_feats.at[: self.num_agents, 2].set(1)
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)
        node_feats = node_feats.at[-n_hits:, 0].set(1)

        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(DoubleIntegrator.GOAL)
        node_type = node_type.at[-n_hits:].set(DoubleIntegrator.OBS)

        lidar_feat = state.obs_history[-1, :, 6:].reshape(
            self.num_agents, self._params["n_rays"], 4
        )
        dist = jnp.maximum(lidar_feat[:, :, 1], 0.0) * self._params["comm_radius"]
        rel_lidar = jnp.stack(
            [dist * lidar_feat[:, :, 2], dist * lidar_feat[:, :, 3]], axis=-1
        )
        abs_lidar = rel_lidar + state.agent[:, None, :2]
        lidar_xy  = merge01(abs_lidar)
        lidar_data = jnp.concatenate([lidar_xy, jnp.zeros_like(lidar_xy)], axis=-1)
        edge_blocks = self.edge_blocks(state, lidar_data)

        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal, lidar_data], axis=0),
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """Lower and upper limits of the agent state."""
        lower_lim = jnp.array([-jnp.inf, -jnp.inf, -0.5, -0.5])
        upper_lim = jnp.array([jnp.inf, jnp.inf, 0.5, 0.5])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        """Lower and upper limits of the agent action."""
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim

    def u_nom_from_state(self, agent_states: jnp.ndarray, goal_states: jnp.ndarray) -> jnp.ndarray:
        """Nominal LQR action from raw state arrays (no GraphsTuple required)."""
        error = goal_states - agent_states
        norm = jnp.linalg.norm(error, axis=-1, keepdims=True) + 1e-6
        error_max = jnp.abs(error / norm * self._params["comm_radius"])
        error = jnp.clip(error, -error_max, error_max)
        return self.clip_action(error @ self._K.T)

    def u_ref(self, graph: GraphsTuple) -> Action:
        """Nominal (reference) action for each agent."""
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        return self.u_nom_from_state(agent, goal)

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        """Graph obtained by applying the action to the current graph."""
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states  = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states   = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)

        next_obs = self._get_obs(next_agent_states, goal_states, graph.env_states.obstacle)
        new_obs_history = jnp.concatenate(
            [graph.env_states.obs_history[1:], next_obs[None, ...]], axis=0
        )
        new_action_history = jnp.concatenate(
            [graph.env_states.action_history[1:], action[None, ...]], axis=0
        )
        next_env_state = graph.env_states._replace(
            agent=next_agent_states,
            obs_history=new_obs_history,
            action_history=new_action_history,
            step_count=graph.env_states.step_count + 1,
        )

        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)
        next_graph  = graph._replace(env_states=next_env_state)
        return self.add_edge_feats(next_graph, next_states)

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, graph: GraphsTuple) -> Array:
        """Mask of agents that are clearly safe."""
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)
        safe_agent = jnp.greater(dist, self._params["car_radius"] * 4)

        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle,
                             self._params["car_radius"] * 2 * self._params.get("safe_buffer_mult", 1.0))
        )

        safe_mask = jnp.logical_and(safe_agent, safe_obs)

        return safe_mask

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        """Mask of agents that are in collision or about to collide."""
        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_state[:, :2]

        agent_pos_diff = agent_pos[None, :, :] - agent_pos[:, None, :]
        agent_dist = jnp.linalg.norm(agent_pos_diff, axis=-1)
        agent_dist = agent_dist + jnp.eye(agent_dist.shape[1]) * (self._params["car_radius"] * 2 + 1)
        unsafe_agent = jnp.less(agent_dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle,
                                      self._params["car_radius"] * self._params.get("safe_buffer_mult", 1.0))

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        agent_warn_dist = 3 * self._params["car_radius"]
        obs_warn_dist = 2 * self._params["car_radius"] * self._params.get("safe_buffer_mult", 1.0)
        obs_pos = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)[:, :2]
        obs_pos_diff = obs_pos[None, :, :] - agent_pos[:, None, :]
        obs_dist = jnp.linalg.norm(obs_pos_diff, axis=-1)
        pos_diff = jnp.concatenate([agent_pos_diff, obs_pos_diff], axis=1)
        warn_zone = jnp.concatenate([jnp.less(agent_dist, agent_warn_dist), jnp.less(obs_dist, obs_warn_dist)], axis=1)
        pos_vec = (pos_diff / (jnp.linalg.norm(pos_diff, axis=2, keepdims=True) + 0.0001))
        speed_agent = jnp.linalg.norm(agent_state[:, 2:], axis=1, keepdims=True)
        heading_vec0 = (agent_state[:, 2:] / (speed_agent + 0.0001))[:, None, :]
        heading_vec = heading_vec0.repeat(pos_vec.shape[1], axis=1)
        inner_prod = jnp.sum(pos_vec * heading_vec, axis=2)
        unsafe_theta_agent = jnp.arctan2(self._params['car_radius'] * 2,
                                         jnp.sqrt(agent_dist**2 - 4 * self._params['car_radius']**2))
        unsafe_theta_obs = jnp.arctan2(self._params['car_radius'],
                                       jnp.sqrt(obs_dist**2 - self._params['car_radius']**2))
        unsafe_theta = jnp.concatenate([unsafe_theta_agent, unsafe_theta_obs], axis=1)
        lidar_mask = jnp.ones((self._params["n_rays"],))
        lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * self.num_agents)
        valid_mask = jnp.concatenate([jnp.ones((self.num_agents, self.num_agents)), lidar_mask], axis=-1)
        warn_zone = jnp.logical_and(warn_zone, valid_mask)
        unsafe_dir = jnp.max(jnp.logical_and(warn_zone, jnp.greater(inner_prod, jnp.cos(unsafe_theta))), axis=1)

        return jnp.logical_or(collision_mask, unsafe_dir)

    def collision_mask(self, graph: GraphsTuple) -> Array:
        """Mask of agents currently colliding with an obstacle."""
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return collision_mask

    def finish_mask(self, graph: GraphsTuple) -> Array:
        """Mask of agents that have reached their goal."""
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 2
        return reach

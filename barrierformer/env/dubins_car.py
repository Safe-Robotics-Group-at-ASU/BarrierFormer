"""
Dubins Car environment (BarrierFormer).

A unicycle with state x = [px, py, theta, v] and control u = [omega, a],
integrated with forward Euler at dt = 0.03 s (paper Eq. 19).  A single agent
navigates to a goal through randomly placed rectangular obstacles, sensing them
only through a LiDAR ring of `n_rays` beams within `comm_radius`.

Same structure as `double_integrator.py` - see that file for the shared design.
What differs here:

1. Nonlinear dynamics - heading theta couples the controls to world-frame motion,
   so the SQP teacher linearises a genuinely nonlinear transition.
2. Nominal controller - `u_nom_from_state_alt` (a heading + speed regulator with
   shortest-angle wrapping), NOT the LQR used by Double Integrator.  It is the
   only nominal controller here; see `u_nom_from_state`.
3. `agent_step_euler` takes a `stop_mask` - agents that reach the goal freeze,
   and their actions are excluded from the reward.

Dynamics entry points, as in Double Integrator:

    agent_step_euler   clips the state, takes stop_mask.  Used by env.step().
    step_simulator     clips, no stop_mask.  Used by the SQP teacher and as the
                       loss_dyn target.
"""

import functools as ft
import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from typing import NamedTuple, Tuple, Optional

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, AgentState, Array, Cost, Done, Info, Pos2d, Reward, State
from ..utils.utils import merge01, jax_vmap
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
from .plot import render_video
from .utils import get_lidar, inside_obstacles, get_node_goal_rng


class DubinsCar(MultiAgentEnv):
    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        agent: AgentState
        goal: State
        obstacle: Obstacle
        obs_history: Array      # [history_len, n_agents, obs_dim]
        action_history: Array   # [history_len-1, n_agents, act_dim]
        step_count: Array

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 16,
        "obs_len_range": [0.1, 0.6],
        "n_obs": 8,
    }

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.03,
            params: dict = None,
            history_len: int = 12,
    ):
        super(DubinsCar, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self.create_obstacles = jax.vmap(Rectangle.create)
        self.enable_stop = True
        self.history_len = history_len
        n_rays = self._params["n_rays"]
        _thetas = np.linspace(-np.pi, np.pi - 2 * np.pi / n_rays, n_rays)
        self._ray_cos = jnp.array(np.cos(_thetas))  # (n_rays,)
        self._ray_sin = jnp.array(np.sin(_thetas))  # (n_rays,)


    # ========================================================================
    # DIMENSIONS
    #   obs_dim = 134 for n_rays=32 (paper Table 3).
    # ========================================================================
    @property
    def state_dim(self) -> int:
        return 4  # x, y, theta, v

    @property
    def node_dim(self) -> int:
        return 3  # indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_rel, vy_rel

    @property
    def action_dim(self) -> int:
        return 2  # omega, acc

    @property
    def obs_dim(self) -> int:
        # [x,y,theta,v](4) + rel_goal(2) + Way-2 lidar (n_rays*4)
        return self.state_dim + 2 + self._params["n_rays"] * 4


    # ========================================================================
    # OBSERVATION MODEL  (o_t, paper Eq. 2)
    #   state(4) + rel_goal(2) + n_rays x 4 LiDAR features; 134-dim at n_rays=32.
    # ========================================================================
    def _get_obs(self, agent_states: AgentState, goal_states: State, obstacles) -> Array:
        """
        Build per-agent observation (Way-2, obs_dim-dimensional):
            o_t = [ x_t(4) | r_t^goal(2) | (hit_mask, dist_norm, cos_θ, sin_θ)×n_rays ]

        Identical encoding to DoubleIntegrator._get_obs — only agent_states semantics differ
        ([x,y,theta,v] vs [x,y,vx,vy]), but position is always agent_states[:,:2].
        """
        sense_range = self._params["comm_radius"]
        get_lidar_fn = jax_vmap(
            ft.partial(
                get_lidar,
                obstacles=obstacles,
                num_beams=self._params["n_rays"],
                sense_range=sense_range,
                max_returns=self._params["n_rays"],
            )
        )
        lidar_xy = get_lidar_fn(agent_states[:, :2])   # (n_agents, n_rays, 2)

        def encode_agent(agent_pos, lidar_hits):
            hits_sg  = jax.lax.stop_gradient(lidar_hits)
            rel      = hits_sg - agent_pos[None, :]
            dist     = jnp.sqrt(jnp.sum(rel ** 2, axis=-1) + 1e-8)
            hit_mask = jnp.less(dist, sense_range).astype(jnp.float32)

            all_sdf  = jax.vmap(lambda obs_j: obs_j.signed_distance(agent_pos))(obstacles)
            min_sdf  = all_sdf.min()
            is_inside = min_sdf < 0.0

            dist_norm = jnp.where(
                is_inside,
                jnp.clip(min_sdf / sense_range, -1.0, 0.0),
                jnp.clip(jnp.where(hit_mask > 0.5, dist / sense_range, 1.0), 0.0, 1.0),
            )
            encoded = jnp.stack(
                [hit_mask, dist_norm, self._ray_cos, self._ray_sin], axis=-1
            )                                           # (n_rays, 4)
            return encoded.reshape(-1)                  # (n_rays*4,)

        lidar_encoded = jax.vmap(encode_agent)(
            agent_states[:, :2], lidar_xy)              # (n_agents, n_rays*4)
        rel_goal = goal_states[:, :2] - agent_states[:, :2]   # (n_agents, 2)
        return jnp.concatenate(
            [agent_states, rel_goal, lidar_encoded], axis=-1)  # (n_agents, obs_dim)


    # ========================================================================
    # RESET  (sample obstacles / start / goal, init history buffers)
    #   Heading is sampled uniformly; goal heading points start -> goal.
    # ========================================================================
    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # randomly generate obstacles
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (self._params["n_obs"], 2), minval=0, maxval=self.area_size)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (self._params["n_obs"], 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (self._params["n_obs"],), minval=0, maxval=2 * np.pi)
        obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, obstacles, self.num_agents, 4 * self.params["car_radius"], self.max_travel)

        # add random heading
        theta_key, key = jr.split(key, 2)
        states = jnp.concatenate([states, jnp.zeros((self.num_agents, 2))], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, 2))], axis=1)
        states = states.at[:, 2].set(jr.uniform(theta_key, (self.num_agents,), minval=-np.pi, maxval=np.pi))
        goals = goals.at[:, 2].set(jnp.arctan2(goals[:, 1] - states[:, 1], goals[:, 0] - states[:, 0]))

        initial_obs = self._get_obs(states, goals, obstacles)   # (n_agents, obs_dim)
        obs_history = jnp.repeat(initial_obs[None, ...], self.history_len, axis=0)
        action_history = jnp.zeros((self.history_len - 1, self.num_agents, self.action_dim))

        env_states = self.EnvState(
            agent=states,
            goal=goals,
            obstacle=obstacles,
            obs_history=obs_history,
            action_history=action_history,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

        return self.get_graph(env_states)


    # ========================================================================
    # DYNAMICS  (paper Eq. 19, nonlinear unicycle)
    #   agent_step_euler takes stop_mask; step_simulator does not.
    # ========================================================================
    def agent_step_euler(self, agent_states: AgentState, action: Action, stop_mask: Array) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = self.agent_xdot(agent_states, action) * (1 - stop_mask)[:, None]
        n_state_agent_new = agent_states + x_dot * self.dt
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.concatenate([
            (jnp.cos(agent_states[:, 2]) * agent_states[:, 3])[:, None],
            (jnp.sin(agent_states[:, 2]) * agent_states[:, 3])[:, None],
            (action[:, 0] * 20.)[:, None],
            (action[:, 1])[:, None]
        ], axis=1)
        assert x_dot.shape == (self.num_agents, self.state_dim)
        return x_dot

    def step_simulator(self, agent_states: AgentState, action: Action) -> AgentState:
        """Pure JAX Euler step without stop_mask — used by SQP and horizon rollout.

        Signature: (n_agents, 4), (n_agents, 2) → (n_agents, 4)
        JAX-differentiable through agent_xdot and clip_state.
        """
        x_dot = self.agent_xdot(agent_states, action)
        return self.clip_state(agent_states + x_dot * self.dt)


    # ========================================================================
    # ENVIRONMENT STEP  (real bookkeeping + history update)
    #   Advances state, appends (o_t, u_t) to the transformer history window.
    # ========================================================================
    def step(
            self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        stop_mask = self.stop_mask(graph)
        if not self.enable_stop:
            # If stopping is not enabled, then set stop_mask to always be 0.
            stop_mask = 0 * stop_mask
        next_agent_states = self.agent_step_euler(agent_states, action, stop_mask)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        # Mask out stopped agents: once an agent reaches the goal (stop_mask=1)
        # its physics are frozen, so penalising its action is meaningless and
        # biases training by punishing the policy for post-goal timesteps.
        reward = jnp.zeros(()).astype(jnp.float32)
        active = (1.0 - stop_mask)                                          # 0 if at goal
        reward -= (active * jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        cost = self.get_cost(graph)

        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        next_obs = self._get_obs(next_agent_states, goal_states, obstacles)
        obs_history = graph.env_states.obs_history
        action_history = graph.env_states.action_history
        new_obs_history = jnp.concatenate([obs_history[1:], next_obs[None, ...]], axis=0)
        new_action_history = jnp.concatenate([action_history[1:], action[None, ...]], axis=0)

        next_state = self.EnvState(
            agent=next_agent_states,
            goal=goal_states,
            obstacle=obstacles,
            obs_history=new_obs_history,
            action_history=new_action_history,
            step_count=graph.env_states.step_count + 1,
        )

        return self.get_graph(next_state), reward, cost, done, {}

    def get_cost(self, graph: EnvGraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        collision = (self._params["car_radius"] * 2 > dist).any(axis=1)
        cost = collision.mean()

        # collision between agents and obstacles
        collision = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])
        cost += collision.mean()

        return cost

    def render_video(
        self, rollout: RolloutResult, video_path: pathlib.Path, Ta_is_unsafe=None, viz_opts: dict = None, dpi: int = 80, **kwargs
    ) -> None:
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


    # ========================================================================
    # GRAPH CONSTRUCTION  (inherited GCBF+ plumbing)
    #   BarrierFormer is single-agent; kept so the GCBF+ tooling still works.
    # ========================================================================
    def edge_blocks(self, state: EnvState, lidar_data: State) -> list[EdgeBlock]:
        n_hits = self._params["n_rays"] * self.num_agents

        # agent - agent connection
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        pos_theta_diff = state.agent[:, None, :2] - state.agent[None, :, :2]
        agent_v = jnp.concatenate([(state.agent[:, 3] * jnp.cos(state.agent[:, 2]))[:, None],
                                   (state.agent[:, 3] * jnp.sin(state.agent[:, 2]))[:, None]], axis=-1)
        v_diff = agent_v[:, None, :] - agent_v[None, :, :]
        state_diff = jnp.concatenate([pos_theta_diff, v_diff], axis=-1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection
        agent_goal_edges = []
        agent_goal_pos_diff = state.agent[:, :2] - state.goal[:, :2]
        agent_goal_v_diff = agent_v
        agent_goal_edge_feats = jnp.concatenate([agent_goal_pos_diff, agent_goal_v_diff], axis=-1)
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_edge_feats = agent_goal_edge_feats.at[:, :2].set(agent_goal_edge_feats[:, :2] * coef)
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        for i in range(self.num_agents):
            agent_goal_edges.append(
                EdgeBlock(agent_goal_edge_feats[i][None, None, :], jnp.ones((1, 1)), id_agent[i][None], id_goal[i][None]))

        # agent - obs connection
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self._params["n_rays"], (i + 1) * self._params["n_rays"])
            lidar_pos = agent_pos[i, :] - lidar_data[id_hits, :2]
            lidar_feats = jnp.concatenate([state.agent[i, :2], agent_v[i]]) - lidar_data[id_hits, :]
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            agent_obs_mask = jnp.ones((1, self._params["n_rays"]))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        f = jnp.concatenate([
            (jnp.cos(state[:, 2]) * state[:, 3])[:, None],
            (jnp.sin(state[:, 2]) * state[:, 3])[:, None],
            jnp.zeros((state.shape[0], 2))
        ], axis=1)
        g = jnp.concatenate([jnp.zeros((2, 2)), jnp.array([[10., 0.], [0., 1.]])], axis=0)
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], 4, 2)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        v = jnp.concatenate([(state[:, 3] * jnp.cos(state[:, 2]))[:, None],
                             (state[:, 3] * jnp.sin(state[:, 2]))[:, None]], axis=-1)
        edge_state = jnp.concatenate([state[:, :2], v], axis=-1)
        assert edge_state.shape[1] == self.edge_dim
        edge_feats = edge_state[graph.receivers] - edge_state[graph.senders]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :2].set(edge_feats[:, :2] * coef)
        return graph._replace(edges=edge_feats, states=state)

    def get_graph(self, state: EnvState, adjacency: Array = None) -> GraphsTuple:
        # node features
        n_hits = self._params["n_rays"] * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, 3))
        node_feats = node_feats.at[: self.num_agents, 2].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)  # goal feats
        node_feats = node_feats.at[-n_hits:, 0].set(1)  # obs feats

        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(DubinsCar.GOAL)
        node_type = node_type.at[-n_hits:].set(DubinsCar.OBS)

        # Recover lidar hit positions from the last obs_history slot (Way-2 encoding).
        # obs_history[-1] layout: [state(4) | rel_goal(2) | (hit,dist_norm,cos,sin)×n_rays]
        # Relative hit xy: rel_x = dist_norm*R*cos_θ,  rel_y = dist_norm*R*sin_θ
        lidar_feat = state.obs_history[-1, :, 6:].reshape(
            self.num_agents, self._params["n_rays"], 4
        )                                                            # (n_agents, n_rays, 4)
        dist_map = jnp.maximum(lidar_feat[:, :, 1], 0.0) * self._params["comm_radius"]
        rel_lidar = jnp.stack(
            [dist_map * lidar_feat[:, :, 2], dist_map * lidar_feat[:, :, 3]], axis=-1
        )                                                            # (n_agents, n_rays, 2)
        abs_lidar = rel_lidar + state.agent[:, None, :2]            # (n_agents, n_rays, 2)
        lidar_xy  = merge01(abs_lidar)                              # (n_agents*n_rays, 2)
        lidar_data = jnp.concatenate([lidar_xy, jnp.zeros_like(lidar_xy)], axis=-1)
        edge_blocks = self.edge_blocks(state, lidar_data)

        # create graph
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal, lidar_data], axis=0),
        ).to_padded()


    # ========================================================================
    # STATE / ACTION BOUNDS
    #   State: v in [-0.8, 0.8] (position and heading are unbounded).
    #   Action: both channels of u = [u_omega, a] are bounded to [-3, 3].
    #
    # ========================================================================
    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[State, State],
            limits of the state
        """
        lower_lim = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -0.8])
        upper_lim = jnp.array([jnp.inf, jnp.inf, jnp.inf, 0.8])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[Action, Action],
            limits of the action
        """
        lower_lim = jnp.ones(2) * -3.0
        upper_lim = jnp.ones(2) * 3.0
        return lower_lim, upper_lim


    # ========================================================================
    # NOMINAL CONTROLLER  (mu_nom, paper Sec. 4)
    #   Heading + speed regulator. The actor head learns u = mu_nom(x) + delta_u.
    # ========================================================================
    def u_nom_from_state(self, agent_states: jnp.ndarray, goal_states: jnp.ndarray) -> Action:
        """Nominal controller mu_nom the actor head corrects (paper Sec. 4).

        This is the ONLY nominal controller for DubinsCar.  An earlier
        multi-branch PID inherited from GCBF+ (`u_nom_from_state_orig`) was
        selected by a `use_orig_u_ref` param that nothing ever set, so it was
        unreachable; it has been removed and `_alt` below is the implementation
        that produced every DubinsCar result in the paper.
        """
        return self.u_nom_from_state_alt(agent_states, goal_states)

    def u_nom_from_state_alt(self, agent_states: jnp.ndarray, goal_states: jnp.ndarray) -> Action:
        """Nominal controller with shortest-angle heading error.

        Design notes (vs. the removed GCBF+ PID):
        - arctan2(sin, cos) wraps heading error to [-pi, pi] without multi-branch logic
        - omega divided by 20 to account for agent_xdot's 20x multiplier;
          k_omega=10 -> effective gain = 10*0.03 = 0.3/step (fast, stable, < 1)
        - velocity target uses raw dist (not comm_radius-capped), so the agent
          can traverse the full 4 m arena within the 256-step budget
        """
        pos_diff = goal_states[:, :2] - agent_states[:, :2]
        dist = jnp.linalg.norm(pos_diff, axis=-1)

        target_theta = jnp.arctan2(pos_diff[:, 1], pos_diff[:, 0])
        current_theta = agent_states[:, 2]
        theta_err = jnp.arctan2(
            jnp.sin(target_theta - current_theta),
            jnp.cos(target_theta - current_theta),
        )

        # effective turn gain = k_omega * dt = 10.0 * 0.03 = 0.3/step
        k_omega = 10.0
        omega_action = (k_omega * theta_err) / 20.0

        # velocity target scales with distance, capped at state_lim (0.8 m/s)
        k_v = 1.0
        k_a = 2.5
        v_target = jnp.clip(k_v * dist, 0.0, 0.8)
        accel_action = k_a * (v_target - agent_states[:, 3])

        action = jnp.concatenate([omega_action[:, None], accel_action[:, None]], axis=-1)
        return self.clip_action(action)

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal  = graph.type_states(type_idx=1, n_type=self.num_agents)
        return self.u_nom_from_state(agent, goal)

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states  = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states   = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        stop_mask = self.stop_mask(graph)
        next_agent_states = self.agent_step_euler(agent_states, action, stop_mask)

        next_obs = self._get_obs(next_agent_states, goal_states, graph.env_states.obstacle)
        new_obs_history = jnp.concatenate(
            [graph.env_states.obs_history[1:], next_obs[None, ...]], axis=0)
        new_action_history = jnp.concatenate(
            [graph.env_states.action_history[1:], action[None, ...]], axis=0)
        next_env_state = graph.env_states._replace(
            agent=next_agent_states,
            obs_history=new_obs_history,
            action_history=new_action_history,
            step_count=graph.env_states.step_count + 1,
        )

        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)
        next_graph  = graph._replace(env_states=next_env_state)
        return self.add_edge_feats(next_graph, next_states)


    # ========================================================================
    # SAFETY LABELS  (D_S / D_U, paper Sec. 4 'Data collection and labeling')
    #   Feed the barrier critic's classification loss.
    # ========================================================================
    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are not colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        safe_agent = jnp.greater(dist, self._params["car_radius"] * 4)

        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 2)
        )

        safe_mask = jnp.logical_and(safe_agent, safe_obs)

        return safe_mask

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_state[:, :2]

        # agents are colliding
        agent_pos_diff = agent_pos[None, :, :] - agent_pos[:, None, :]
        agent_dist = jnp.linalg.norm(agent_pos_diff, axis=-1)
        agent_dist = agent_dist + jnp.eye(agent_dist.shape[1]) * (self._params["car_radius"] * 2 + 1)
        unsafe_agent = jnp.less(agent_dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 1.5)

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        # unsafe direction
        agent_warn_dist = 3 * self._params["car_radius"]
        obs_warn_dist = 2 * self._params["car_radius"]
        obs_pos = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)[:, :2]
        obs_pos_diff = obs_pos[None, :, :] - agent_pos[:, None, :]
        obs_dist = jnp.linalg.norm(obs_pos_diff, axis=-1)
        pos_diff = jnp.concatenate([agent_pos_diff, obs_pos_diff], axis=1)
        warn_zone = jnp.concatenate([jnp.less(agent_dist, agent_warn_dist), jnp.less(obs_dist, obs_warn_dist)], axis=1)
        pos_vec = (pos_diff / (jnp.linalg.norm(pos_diff, axis=2, keepdims=True) + 0.0001))
        heading_vec = jnp.concatenate([jnp.cos(agent_state[:, 2])[:, None],
                                       jnp.sin(agent_state[:, 2])[:, None]], axis=1)[:, None, :]
        heading_vec = heading_vec.repeat(pos_vec.shape[1], axis=1)
        inner_prod = jnp.sum(pos_vec * heading_vec, axis=2)
        unsafe_theta_agent = jnp.arctan2(self._params['car_radius'] * 2,
                                         jnp.sqrt(agent_dist ** 2 - 4 * self._params['car_radius'] ** 2))
        unsafe_theta_obs = jnp.arctan2(self._params['car_radius'],
                                       jnp.sqrt(obs_dist ** 2 - self._params['car_radius'] ** 2))
        unsafe_theta = jnp.concatenate([unsafe_theta_agent, unsafe_theta_obs], axis=1)
        lidar_mask = jnp.ones((self._params["n_rays"],))
        lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * self.num_agents)
        valid_mask = jnp.concatenate([jnp.ones((self.num_agents, self.num_agents)), lidar_mask], axis=-1)
        warn_zone = jnp.logical_and(warn_zone, valid_mask)
        unsafe_dir = jnp.max(jnp.logical_and(warn_zone, jnp.greater(inner_prod, jnp.cos(unsafe_theta))), axis=1)

        return jnp.logical_or(collision_mask, unsafe_dir)


    # ========================================================================
    # EVALUATION MASKS
    #   collision_mask / finish_mask drive the reported metrics; stop_mask freezes
    # ========================================================================
    def collision_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return collision_mask

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 2
        return reach

    def stop_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        stop = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 0.5
        return stop

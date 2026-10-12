import functools as ft
import pathlib
import control as ct
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from typing import NamedTuple, Optional, Tuple

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, AgentState, Array, Cost, Done, Info, Pos3d, Reward, State
from ..utils.utils import merge01, jax_vmap
from .base import MultiAgentEnv, RolloutResult
from .linear_drone import LinearDrone
from .obstacle import Obstacle, Sphere
from .plot import render_video
from .utils import RK4_step, get_lidar, inside_obstacles, get_node_goal_rng


def get_rotmat(phi, theta, psi):
    """Rotation matrix from roll, pitch and yaw."""
    c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
    c_th, s_th = jnp.cos(theta), jnp.sin(theta)
    c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
    R_W_cf = jnp.array(
        [
            [c_psi * c_th, c_psi * s_th * s_phi - s_psi * c_phi, c_psi * s_th * c_phi + s_psi * s_phi],
            [s_psi * c_th, s_psi * s_th * s_phi + c_psi * c_phi, s_psi * s_th * c_phi - c_psi * s_phi],
            [-s_th, c_th * s_phi, c_th * c_phi],
        ]
    )
    return R_W_cf


class CrazyFlie(MultiAgentEnv):
    """CrazyFlie quadrotor environment; actions are velocity targets tracked by an inner LQR."""

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
        "drone_radius": 0.05,
        "comm_radius": 1.0,
        "n_rays": 16,
        "obs_len_range": [0.1, 0.6],
        "n_obs": 0,
        "m": 0.0299,
        "Ixx": 1.395 * 10 ** (-5),
        "Iyy": 1.395 * 10 ** (-5),
        "Izz": 2.173 * 10 ** (-5),
        "CT": 3.1582 * 10 ** (-10),
        "CD": 7.9379 * 10 ** (-12),
        "d": 0.03973,
        "lidar_encoding": "sector",
        "lidar_phantom_eps": 1e-3,
    }

    X, Y, Z, PSI, THETA, PHI, U, V, W, R, Q, P = range(12)

    F_1, F_2, F_3, F_4 = range(4)

    L_PHI, L_THETA, L_PSI, L_P, L_Q, L_R, L_VX, L_VY, L_VZ = range(9)
    L_U_VX, L_U_VY, L_U_VZ, L_U_R = range(4)

    def __init__(
        self,
        num_agents: int,
        area_size: float,
        max_step: int = 256,
        max_travel: float = None,
        dt: float = 0.03,
        params: dict = None,
        history_len: int = 32,
    ):
        """Create the environment with the given parameters."""
        super().__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self.history_len = history_len
        if self._params.get("lidar_encoding", "sector") not in ("sector", "nearest"):
            raise ValueError(f"lidar_encoding={self._params.get('lidar_encoding')!r}; use 'sector' or 'nearest'")
        self._build_sector_layout()

        self.create_obstacles = jax.vmap(Sphere.create)
        self.n_rays = 16
        self.n_rays = min(self.n_rays, self._params["n_rays"] ** 2 // 2 + 2)
        self.normalize_by_CT = True

        self.vel_targets_scale = np.array([2.0, 2.0, 0.5, 0.1])

        self._K_ll = self._compute_K_ll()
        self._K_nom = self._compute_K_nom()

    @property
    def state_dim(self) -> int:
        """Dimension of one agent's state."""
        return 12

    @property
    def node_dim(self) -> int:
        """Dimension of a graph node feature."""
        return 3

    @property
    def edge_dim(self) -> int:
        """Dimension of a graph edge feature."""
        return 12

    @property
    def action_dim(self) -> int:
        """Dimension of one agent's action."""
        return 4

    @property
    def obs_dim(self) -> int:
        """Dimension of the per-agent observation vector."""
        return self.state_dim + 3 + self._params["n_rays"] * 5

    def _build_sector_layout(self):
        """Sector membership of the cast LiDAR rays."""
        nr = self._params["n_rays"]
        n_el, n_az = 4, nr // 4
        th = np.linspace(-np.pi / 2 + 2 * np.pi / nr, np.pi / 2 - 2 * np.pi / nr, nr // 2)
        ph = np.linspace(-np.pi, np.pi - 2 * np.pi / nr, nr)
        assert len(th) % n_el == 0 and len(ph) % n_az == 0
        n_cast = len(th) * len(ph) + 2
        member = np.zeros((n_el * n_az, n_cast), dtype=bool)
        for ti in range(len(th)):
            for pj in range(len(ph)):
                el, az = ti // (len(th) // n_el), pj // (len(ph) // n_az)
                member[el * n_az + az, ti * len(ph) + pj] = True
        member[(n_el - 1) * n_az:, n_cast - 2] = True
        member[:n_az, n_cast - 1] = True
        el_c = np.array([th[e * (len(th) // n_el):(e + 1) * (len(th) // n_el)].mean() for e in range(n_el)])
        az_c = np.array([ph[a * (len(ph) // n_az):(a + 1) * (len(ph) // n_az)].mean() for a in range(n_az)])
        center = np.array([[np.cos(el_c[e]) * np.cos(az_c[a]), np.cos(el_c[e]) * np.sin(az_c[a]), np.sin(el_c[e])]
                           for e in range(n_el) for a in range(n_az)])
        self._sector_member = jnp.asarray(member)
        self._sector_center = jnp.asarray(center, dtype=jnp.float32)

    def _lidar_hits(self, agent_states: AgentState, obstacles) -> Array:
        """3D points for the n_rays LiDAR slots, (n_agents, n_rays, 3)."""
        nr, sense_range = self._params["n_rays"], self._params["comm_radius"]
        eps = self._params["lidar_phantom_eps"]
        if self._params["lidar_encoding"] == "nearest":
            fn = jax_vmap(ft.partial(get_lidar, obstacles=obstacles, num_beams=nr,
                                     sense_range=sense_range, max_returns=nr))
            return fn(agent_states[:, :3])
        fn = jax_vmap(ft.partial(get_lidar, obstacles=obstacles, num_beams=nr,
                                 sense_range=sense_range, max_returns=nr, sort=False))
        all_hits = fn(agent_states[:, :3])

        def per_agent(pos, hits):
            d = jnp.linalg.norm(hits - pos[None], axis=-1)
            real = d < sense_range - eps
            D = jnp.where(self._sector_member & real[None], d[None], jnp.inf)
            idx = jnp.argmin(D, axis=1)
            has = jnp.isfinite(jnp.min(D, axis=1))
            empty = pos[None] + 2.0 * sense_range * self._sector_center
            return jnp.where(has[:, None], hits[idx], empty)

        return jax.vmap(per_agent)(agent_states[:, :3], all_hits)

    def _get_obs(self, agent_states: AgentState, goal_states: State, obstacles) -> Array:
        """Build the per-agent observation [state | goal offset | LiDAR features]."""
        sense_range = self._params["comm_radius"]
        eps = self._params["lidar_phantom_eps"]
        lidar_xyz = self._lidar_hits(agent_states, obstacles)

        def encode_agent(agent_pos3, lidar_hits_a):
            hits_sg = jax.lax.stop_gradient(lidar_hits_a)
            rel = hits_sg - agent_pos3[None, :]
            dist = jnp.sqrt(jnp.sum(rel ** 2, axis=-1) + 1e-8)
            hit_mask = jnp.less(dist, sense_range - eps).astype(jnp.float32)
            all_sdf = jax.vmap(lambda obs_j: jnp.linalg.norm(agent_pos3 - obs_j.center) - obs_j.radius)(obstacles)
            min_sdf = all_sdf.min()
            is_inside = min_sdf < 0.0
            dist_norm = jnp.where(
                is_inside,
                jnp.clip(min_sdf / sense_range, -1.0, 0.0),
                jnp.clip(jnp.where(hit_mask > 0.5, dist / sense_range, 1.0), 0.0, 1.0),
            )
            dir_vec = jax.lax.stop_gradient(rel / (dist[:, None] + 1e-8))
            encoded = jnp.concatenate([hit_mask[:, None], dist_norm[:, None], dir_vec], axis=-1)
            return encoded.reshape(-1)

        lidar_encoded = jax.vmap(encode_agent)(agent_states[:, :3], lidar_xyz)
        rel_goal = goal_states[:, :3] - agent_states[:, :3]
        return jnp.concatenate([agent_states, rel_goal, lidar_encoded], axis=-1)

    USE_ENV_OBS_IN_ROLLOUT = True

    def step_simulator(self, agent_states: AgentState, action: Action) -> AgentState:
        """RK4 step WITHOUT the state clip (SQP gradient; same contract as DI / DubinsCar)."""
        return RK4_step(self.agent_xdot, agent_states, action, self.dt)

    def true_next_state(self, agent_states: AgentState, action: Action, goal_states: State) -> AgentState:
        """Exactly the next agent state env.step produces: clip_action, RK4, clip_state."""
        return self.agent_step_rk4(agent_states, self.clip_action(action))

    def u_nom_from_state(self, agent_states: AgentState, goal_states: State) -> Action:
        """LQR nominal action from raw arrays; identical to u_ref(graph)."""
        return jax_vmap(self.u_ref_inner_single)(agent_states, goal_states)

    def reset(self, key: Array) -> GraphsTuple:
        """Sample a new episode (obstacles, start and goal) and return its graph."""
        self._t = 0

        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 3), minval=0, maxval=self.area_size)

        r_key, key = jr.split(key, 2)
        obs_radius = jr.uniform(r_key, (n_rng_obs,),
                                minval=self._params["obs_len_range"][0] / 2,
                                maxval=self._params["obs_len_range"][1] / 2)
        obstacles = self.create_obstacles(obs_pos, obs_radius)

        states, goals = get_node_goal_rng(
            key, self.area_size, 3, obstacles, self.num_agents, 4 * self.params["drone_radius"], self.max_travel)

        states = jnp.concatenate([states, jnp.zeros((self.num_agents, self.state_dim - 3))], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, self.state_dim - 3))], axis=1)

        initial_obs = self._get_obs(states, goals, obstacles)
        obs_history = jnp.repeat(initial_obs[None, ...], self.history_len, axis=0)
        action_history = jnp.zeros((self.history_len - 1, self.num_agents, self.action_dim))
        env_states = self.EnvState(states, goals, obstacles, obs_history, action_history,
                                   jnp.array(0, dtype=jnp.int32))

        return self.get_graph(env_states)

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

        next_agent_states = self.agent_step_rk4(agent_states, action)

        done = jnp.array(False)

        reward = jnp.zeros(()).astype(jnp.float32)
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        cost = self.get_cost(graph)

        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        next_obs = self._get_obs(next_agent_states, goal_states, obstacles)
        new_obs_history = jnp.concatenate([graph.env_states.obs_history[1:], next_obs[None]], axis=0)
        new_action_history = jnp.concatenate([graph.env_states.action_history[1:], action[None]], axis=0)
        next_state = self.EnvState(next_agent_states, goal_states, obstacles, new_obs_history,
                                   new_action_history, graph.env_states.step_count + 1)

        info = {}
        if get_eval_info:
            agent_pos = agent_states[:, :3]
            info["inside_obstacles"] = inside_obstacles(agent_pos, obstacles, r=self._params["drone_radius"])

        return self.get_graph(next_state), reward, cost, done, info

    def get_cost(self, graph: EnvGraphsTuple) -> Cost:
        """Collision cost of the current graph."""
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle

        agent_pos = agent_states[:, :3]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        collision = (self._params["drone_radius"] * 2 > dist).any(axis=1)
        cost = collision.mean()

        collision = inside_obstacles(agent_pos, obstacles, r=self._params["drone_radius"])
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
            dim=3,
            n_agent=self.num_agents,
            n_rays=self.n_rays,
            r=self.params["drone_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            **kwargs
        )

    def edge_state(self, state: State) -> State:

        """State used for edge features (position and world-frame velocity)."""
        def edge_state_fn_(x: State) -> State:
            pos_W = x[:3]
            u = x[CrazyFlie.U]
            v = x[CrazyFlie.V]
            w = x[CrazyFlie.W]

            R_W_cf = self.get_rotation_mat(x)
            v_Wcf_cf = jnp.array([u, v, w])
            v_Wcf_W = R_W_cf @ v_Wcf_cf

            z_W = R_W_cf[:, 2]

            r = x[CrazyFlie.R]
            q = x[CrazyFlie.Q]
            p = x[CrazyFlie.P]
            omega = jnp.array([p, q, r])
            omega_W = R_W_cf @ omega

            return jnp.concatenate([pos_W, v_Wcf_W, z_W, omega_W], axis=-1)

        return jax.vmap(edge_state_fn_)(state)

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        """Add edge features to a graph from the given states."""
        assert graph.is_single
        assert state.ndim == 2

        edge_state = self.edge_state(state)
        edge_feats = edge_state[graph.receivers] - edge_state[graph.senders]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :3] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :3].set(edge_feats[:, :3] * coef)

        return graph._replace(edges=edge_feats, states=state)

    def edge_blocks(self, state: EnvState, lidar_data: Pos3d) -> list[EdgeBlock]:
        """Build the agent-agent, agent-goal and agent-obstacle edge blocks."""
        n_hits = self.num_agents * self.n_rays

        agent_pos = state.agent[:, :3]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_edge_state = self.edge_state(state.agent)
        state_diff = agent_edge_state[:, None, :] - agent_edge_state[None, :, :]
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.eye(self.num_agents)
        goal_edge_state = self.edge_state(state.goal)
        agent_goal_feats = agent_edge_state[:, None, :] - goal_edge_state[None, :, :]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :3] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :3].set(agent_goal_feats[:, :3] * coef)
        agent_goal_edges = EdgeBlock(agent_goal_feats, agent_goal_mask, id_agent, id_goal)

        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        lidar_edge_state = self.edge_state(lidar_data)
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self.n_rays, (i + 1) * self.n_rays)
            lidar_pos = agent_pos[i, :] - lidar_data[id_hits, :3]
            lidar_feats = agent_edge_state[i, :] - lidar_edge_state[id_hits, :]
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            agent_obs_mask = jnp.ones((1, self.n_rays))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges

    def _single_agent_f(self, x: Array):
        """Drift term f(x) of the quadrotor dynamics."""
        assert x.shape == (self.state_dim,)
        Ixx, Iyy, Izz = self._params["Ixx"], self._params["Iyy"], self._params["Izz"]
        I = jnp.array([Ixx, Iyy, Izz])

        phi, theta, psi = x[CrazyFlie.PHI], x[CrazyFlie.THETA], x[CrazyFlie.PSI]

        c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
        c_th, s_th = jnp.cos(theta), jnp.sin(theta)
        c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
        t_th = jnp.tan(theta)

        u, v, w = x[CrazyFlie.U], x[CrazyFlie.V], x[CrazyFlie.W]
        uvw = jnp.array([u, v, w])

        p, q, r = x[CrazyFlie.P], x[CrazyFlie.Q], x[CrazyFlie.R]
        pqr = jnp.array([p, q, r])

        R_W_cf = self.get_rotation_mat(x)
        v_Wcf_cf = jnp.array([u, v, w])
        v_Wcf_W = R_W_cf @ v_Wcf_cf
        assert v_Wcf_W.shape == (3,)

        mat = jnp.array(
            [
                [0, s_phi / c_th, c_phi / c_th],
                [0, c_phi, -s_phi],
                [1, s_phi * t_th, c_phi * t_th],
            ]
        )
        deuler_ypr = mat @ pqr

        acc_cf_g = -R_W_cf[2, :] * 9.81
        acc_cf = -jnp.cross(pqr, uvw) + acc_cf_g

        pqr_dot = -jnp.cross(pqr, I * pqr) / I
        rpq_dot = pqr_dot[::-1]
        assert pqr_dot.shape == (3,)

        x_dot = jnp.concatenate([v_Wcf_W, deuler_ypr, acc_cf, rpq_dot], axis=0)
        assert x_dot.shape == (self.state_dim,)

        return x_dot

    def _single_agent_gu(self, x: Array, control: Array) -> Array:
        """Control term g(x) u of the quadrotor dynamics."""
        assert x.shape == (self.state_dim,)

        params = self._params
        m, Ixx, Iyy, Izz = params["m"], params["Ixx"], params["Iyy"], params["Izz"]
        CT, CD, d = params["CT"], params["CD"], params["d"]

        if self.normalize_by_CT:
            CT, CD = 1, CD / CT

        w_term = jnp.sum(control)
        p_term = jnp.sum(control * jnp.array([-1.0, -1.0, 1.0, 1.0]))
        q_term = jnp.sum(control * jnp.array([-1.0, 1.0, 1.0, -1.0]))
        r_term = jnp.sum(control * jnp.array([-1.0, 1.0, -1.0, 1.0]))

        w_dot = CT * w_term / m
        p_dot = CT * np.sqrt(2) * d * p_term / Ixx
        q_dot = CT * np.sqrt(2) * d * q_term / Ixx
        r_dot = CD * r_term / Izz

        gu = jnp.zeros(self.state_dim)
        gu = gu.at[CrazyFlie.W].set(w_dot)
        gu = gu.at[CrazyFlie.P].set(p_dot)
        gu = gu.at[CrazyFlie.Q].set(q_dot)
        gu = gu.at[CrazyFlie.R].set(r_dot)

        return gu

    def thrust_from_motor(self):
        """Matrix mapping motor forces to thrust and body torques."""
        params = self._params
        m, Ixx, Iyy, Izz = params["m"], params["Ixx"], params["Iyy"], params["Izz"]
        CT, CD, d = params["CT"], params["CD"], params["d"]
        if self.normalize_by_CT:
            CT, CD = 1, CD / CT
        dw_du = CT * np.full(4, 1 / m)
        dp_du = CT * np.sqrt(2) * d * jnp.array([-1.0, -1.0, 1.0, 1.0]) / Ixx
        dq_du = CT * np.sqrt(2) * d * jnp.array([-1.0, 1.0, 1.0, -1.0]) / Iyy
        dr_du = CD * np.array([-1.0, 1.0, -1.0, 1.0]) / Izz

        mat = np.stack([dw_du, dp_du, dq_du, dr_du], axis=0)
        return mat

    def _agent_xdot_single_agent(self, state: AgentState, control: Action) -> AgentState:
        """Low-level quadrotor dynamics x_dot for one agent."""
        f = self._single_agent_f(state)
        gu = self._single_agent_gu(state, control)
        assert f.shape == (self.state_dim,) and gu.shape == (self.state_dim,)
        assert control.shape == (self.action_dim,)
        xdot = f + gu
        assert xdot.shape == (self.state_dim,)
        return xdot

    def _xdot_ll(self, x: AgentState, u: Action) -> AgentState:
        """Dynamics of the reduced low-level state used for the inner LQR."""
        assert x.shape == (9,)
        assert u.shape == (4,)
        m, Ixx, Iyy, Izz = self._params["m"], self._params["Ixx"], self._params["Iyy"], self._params["Izz"]
        I = jnp.array([Ixx, Iyy, Izz])

        CT, CD, d = self._params["CT"], self._params["CD"], self._params["d"]
        if self.normalize_by_CT:
            CT, CD = 1, CD / CT

        phi, theta, psi = x[CrazyFlie.L_PHI], x[CrazyFlie.L_THETA], x[CrazyFlie.L_PSI]

        c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
        c_th, s_th = jnp.cos(theta), jnp.sin(theta)
        c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
        t_th = jnp.tan(theta)

        vx, vy, vz = x[CrazyFlie.L_VX], x[CrazyFlie.L_VY], x[CrazyFlie.L_VZ]
        v_Wcf_W = jnp.array([vx, vy, vz])

        p, q, r = x[CrazyFlie.L_P], x[CrazyFlie.L_Q], x[CrazyFlie.L_R]
        pqr = jnp.array([p, q, r])

        mat = jnp.array(
            [
                [1, s_phi * t_th, c_phi * t_th],
                [0, c_phi, -s_phi],
                [0, s_phi / c_th, c_phi / c_th],
            ]
        )
        deuler_rpy = mat @ pqr

        R_W_cf = get_rotmat(x[CrazyFlie.L_PHI], x[CrazyFlie.L_THETA], x[CrazyFlie.L_PSI])

        acc_W = jnp.array([0.0, 0.0, -9.81])

        pqr_dot = -jnp.cross(pqr, I * pqr) / I
        assert pqr_dot.shape == (3,)

        dw_du = CT * jnp.full(4, 1 / m)
        dp_du = CT * np.sqrt(2) * d * jnp.array([-1.0, -1.0, 1.0, 1.0]) / Ixx
        dq_du = CT * np.sqrt(2) * d * jnp.array([-1.0, 1.0, 1.0, -1.0]) / Iyy
        dr_du = CD * jnp.array([-1.0, 1.0, -1.0, 1.0]) / Izz
        assert dw_du.shape == dp_du.shape == dq_du.shape == dr_du.shape == (self.action_dim,)

        wdot = dw_du @ u
        pdot = dp_du @ u
        qdot = dq_du @ u
        rdot = dr_du @ u

        pqr_dot_control = jnp.array([pdot, qdot, rdot])
        acc_cf_control = jnp.array([0.0, 0.0, wdot])
        acc_W_control = R_W_cf @ acc_cf_control

        x_dot = jnp.concatenate([deuler_rpy, pqr_dot + pqr_dot_control, acc_W + acc_W_control], axis=0)
        assert x_dot.shape == (9,)

        return x_dot

    def _compute_K_ll(self):
        """Inner-loop LQR gain for velocity tracking."""
        def xdot(x, u):
            return self._xdot_ll(x, u + self.u_eq)

        x_zero = np.zeros(9)
        u_zero = np.zeros(4)

        xdot_zero = np.array(xdot(x_zero, u_zero))
        np.testing.assert_allclose(xdot_zero, 0, atol=5e-5)

        A_ll, B_ll = jax.jacobian(xdot, argnums=(0, 1))(x_zero, u_zero)

        A_ll = np.delete(np.delete(A_ll, CrazyFlie.L_PSI, axis=0), CrazyFlie.L_PSI, axis=1)
        assert A_ll.shape == (8, 8)

        B_ll = np.delete(B_ll, CrazyFlie.L_PSI, axis=0)
        assert B_ll.shape == (8, 4)

        Q = 1.0 * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 20.0])
        Q = np.diag(Q)

        R_thrust = 0.01 * np.array([5.0, 1.0, 1.0, 1.0])
        T_fr_M = self.thrust_from_motor()
        R_motor = T_fr_M.T @ np.diag(R_thrust) @ T_fr_M

        K, S, E = ct.lqr(A_ll, B_ll, Q, R_motor)
        assert K.shape == (4, 8)

        K = np.insert(K, CrazyFlie.L_PSI, 0, axis=1)
        assert K.shape == (4, 9)

        return K

    def _compute_K_nom(self):
        """LQR gain of the nominal goal-reaching controller."""
        x_zero, u_zero = np.zeros(self.state_dim), np.zeros(self.action_dim)
        A_hl, B_hl = jax.jacobian(self._agent_xdot_single_agent_hl, argnums=(0, 1))(x_zero, u_zero)

        Q = 2 * np.array([50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        R = 4 * np.array([1.0, 1.0, 1.0, 1.0])

        K, S, E = ct.lqr(A_hl, B_hl, np.diag(Q), np.diag(R))
        return K

    def u_ref_inner_single(self, state: AgentState, goal: AgentState) -> Action:
        """Nominal action for one agent toward its goal."""
        error = jnp.array(state - goal)
        dist = jnp.linalg.norm(error[:3])
        dist_safe = jnp.maximum(dist, 1e-4)
        clip_coef = jnp.where(dist > self.comm_radius, self.comm_radius / dist_safe, 1.0)
        error = error.at[:3].multiply(clip_coef)
        u_hl = -self._K_nom @ error
        return self.clip_action(u_hl)

    @property
    def u_eq(self):
        """Equilibrium (hover) motor input."""
        u_eq = jnp.zeros((self.action_dim,))
        u_eq = u_eq.at[CrazyFlie.F_1].set(self._params["m"] * 9.81 / 4)
        u_eq = u_eq.at[CrazyFlie.F_2].set(self._params["m"] * 9.81 / 4)
        u_eq = u_eq.at[CrazyFlie.F_3].set(self._params["m"] * 9.81 / 4)
        u_eq = u_eq.at[CrazyFlie.F_4].set(self._params["m"] * 9.81 / 4)

        if not self.normalize_by_CT:
            u_eq = u_eq / self._params["CT"]

        return u_eq

    def _get_ll_state(self, state: AgentState) -> AgentState:
        """Reduced low-level state used by the inner LQR."""
        u, v, w = state[CrazyFlie.U], state[CrazyFlie.V], state[CrazyFlie.W]
        uvw = jnp.array([u, v, w])

        phi, theta, psi = state[CrazyFlie.PHI], state[CrazyFlie.THETA], state[CrazyFlie.PSI]
        R_W_cf = self.get_rotation_mat(state)
        v_Wcf_cf = uvw
        v_Wcf_W = R_W_cf @ v_Wcf_cf
        assert v_Wcf_W.shape == (3,)
        vx, vy, vz = v_Wcf_W

        p, q, r = state[CrazyFlie.P], state[CrazyFlie.Q], state[CrazyFlie.R]

        ll_state = jnp.array([phi, theta, psi, p, q, r, vx, vy, vz])
        assert ll_state.shape == (9,)
        return ll_state

    def _vel_targets_to_ll_state(self, vel_targets: Action):
        """Desired low-level state for given velocity targets."""
        vx, vy, vz, r = vel_targets
        ll_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, r, vx, vy, vz])
        assert ll_state.shape == (9,)
        return ll_state

    def _get_ll_controls(self, state: AgentState, vel_targets: Action) -> Action:
        """Motor inputs that track the velocity targets."""
        assert vel_targets.shape == (4,)
        ll_state = self._get_ll_state(state)
        ll_state_des = self._vel_targets_to_ll_state(vel_targets)

        control = -self._K_ll @ (ll_state - ll_state_des)
        control = self.u_eq + control
        return control

    def _agent_xdot_single_agent_hl(self, state: AgentState, vel_targets_scaled: Action) -> AgentState:
        """Closed-loop dynamics for velocity-target actions."""
        assert state.shape == (self.state_dim,)
        assert vel_targets_scaled.shape == (self.action_dim,)

        vel_targets_scaled = self.clip_action(vel_targets_scaled)
        vel_targets = vel_targets_scaled * self.vel_targets_scale

        control = self._get_ll_controls(state, vel_targets)
        return self._agent_xdot_single_agent(state, control)

    def agent_xdot(self, agent_states: AgentState, vel_targets: Action) -> AgentState:
        """Continuous-time agent dynamics x_dot = f(x, u)."""
        assert vel_targets.ndim == agent_states.ndim
        if vel_targets.ndim == 1:
            return self._agent_xdot_single_agent_hl(agent_states, vel_targets)
        return jax_vmap(self._agent_xdot_single_agent_hl)(agent_states, vel_targets)

    def agent_step_rk4(self, agent_state: Array, vel_targets: Array) -> Array:
        """One RK4 step for all agents."""
        assert agent_state.shape == (self.num_agents, self.state_dim)
        assert vel_targets.shape == (self.num_agents, self.action_dim)
        n_state_agent_new = RK4_step(self.agent_xdot, agent_state, vel_targets, self.dt)
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def rk4_single(self, agent_state: Array, vel_targets: Array):
        """One RK4 step for a single agent."""
        assert agent_state.shape == (self.state_dim,)
        assert vel_targets.shape == (self.action_dim,)
        state_agent_new = RK4_step(self._agent_xdot_single_agent_hl, agent_state, vel_targets, self.dt)
        assert state_agent_new.shape == (self.state_dim,)

        control = self._get_ll_controls(agent_state, vel_targets)
        return self.clip_state(state_agent_new), control

    def control_affine_dyn_single(self, state: AgentState) -> tuple[Array, Array]:
        """Control-affine dynamics f(x), g(x) for one agent."""
        u_zero = jnp.zeros(4)
        f = self.agent_xdot(state, u_zero)
        g = jax.jacobian(self.agent_xdot, argnums=1)(state, u_zero)

        return f, g

    def control_affine_dyn(self, state: State) -> tuple[Array, Array]:
        """Control-affine dynamics terms f(x) and g(x)."""
        assert state.ndim == 2
        a_f, a_g = jax_vmap(self.control_affine_dyn_single)(state)
        return a_f, a_g

    @staticmethod
    def get_rotation_mat(x: State) -> State:
        """Rotation matrix of a state."""
        return get_rotmat(x[CrazyFlie.PHI], x[CrazyFlie.THETA], x[CrazyFlie.PSI])

    def action_lim(self) -> Tuple[Action, Action]:
        """Lower and upper limits of the agent action."""
        low_lim = jnp.array([-1.0, -1.0, -1.0, -1.0])
        up_lim = jnp.array([1.0, 1.0, 1.0, 1.0])
        return low_lim, up_lim

    def render(self, graph: GraphsTuple) -> plt.Figure:
        """Not implemented."""
        pass

    def get_graph(self, state: EnvState) -> GraphsTuple:
        """Build the graph for the given environment state."""
        n_hits = self.n_rays * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, 3))
        node_feats = node_feats.at[: self.num_agents, 2].set(1)
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)
        node_feats = node_feats.at[-n_hits:, 0].set(1)

        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(LinearDrone.GOAL)
        node_type = node_type.at[-n_hits:].set(LinearDrone.OBS)

        get_lidar_vmap = jax.vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self.params['n_rays'],
                sense_range=self._params["comm_radius"],
                max_returns=self.n_rays,
            )
        )
        lidar_data = merge01(get_lidar_vmap(state.agent[:, :3]))
        lidar_data = jnp.concatenate(
            [lidar_data, jnp.zeros((lidar_data.shape[0], self.state_dim - lidar_data.shape[1]))], axis=-1
        )
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
        low_lim = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf,
                             -jnp.inf, -jnp.pi / 4, -jnp.pi / 4,
                             -0.3, -0.3, -0.3,
                             -10, -10, -10])
        up_lim = jnp.array([jnp.inf, jnp.inf, jnp.inf,
                            jnp.inf, jnp.pi / 4, jnp.pi / 4,
                            0.3, 0.3, 0.3,
                            10, 10, 10])
        return low_lim, up_lim

    def u_ref(self, graph: GraphsTuple) -> Action:
        """Nominal (reference) action for each agent."""
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        return jax_vmap(self.u_ref_inner_single)(agent, goal)

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        """Graph obtained by applying the action to the current graph."""
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_rk4(agent_states, action)
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)
        return next_graph

    def safe_mask(self, graph: GraphsTuple) -> Array:
        """Mask of agents that are clearly safe."""
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :3]

        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["drone_radius"] * 2 + 1)
        safe_agent = jnp.greater(dist, self._params["drone_radius"] * 4)

        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["drone_radius"] * 2)
        )

        safe_mask = jnp.logical_and(safe_agent, safe_obs)

        return safe_mask

    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        """Mask of agents that are in collision or about to collide."""
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :3]

        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["drone_radius"] * 2 + 1)
        unsafe_agent = jnp.less(dist, self._params["drone_radius"] * 2.5)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["drone_radius"] * 1.5)

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return collision_mask

    def collision_mask(self, graph: GraphsTuple) -> Array:
        """Mask of agents currently colliding with an obstacle."""
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :3]

        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["drone_radius"] * 2 + 1)
        unsafe_agent = jnp.less(dist, self._params["drone_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["drone_radius"])

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return collision_mask

    def finish_mask(self, graph: GraphsTuple) -> Array:
        """Mask of agents that have reached their goal."""
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :3]
        goal_pos = graph.env_states.goal[:, :3]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["drone_radius"] * 2
        return reach

    @property
    def comm_radius(self):
        """Communication (sensing) radius."""
        return self._params["comm_radius"]

    @property
    def x_labels(self):
        """Plot labels of the state components."""
        x_labels = ["x", "y", "z", r"$\psi$", r"$\theta$", r"$\phi$", "u", "v", "w", "r", "q", "p"]
        assert len(x_labels) == self.state_dim
        return x_labels

    @property
    def uhl_labels(self):
        """Plot labels of the action components."""
        uhl_labels = [r"$vx_{tgt}$", r"$vy_{tgt}$", r"$vz_{tgt}$", r"$r_{tgt}$"]
        assert len(uhl_labels) == self.action_dim
        return uhl_labels

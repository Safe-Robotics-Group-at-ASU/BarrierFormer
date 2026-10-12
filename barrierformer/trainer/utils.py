import jax.numpy as jnp
import jax.tree_util as jtu
import jax
import numpy as np
import socket
import matplotlib.pyplot as plt
import functools as ft
import seaborn as sns
import optax

from typing import Callable, TYPE_CHECKING
from matplotlib.colors import CenteredNorm

from ..utils.typing import PRNGKey
from ..utils.graph import GraphsTuple
from .data import Rollout
from ..env.utils import inside_obstacles


if TYPE_CHECKING:
    from ..env import MultiAgentEnv
else:
    MultiAgentEnv = None


def rollout(
        env: MultiAgentEnv,
        actor: Callable,
        key: PRNGKey
) -> Rollout:
    """Get a rollout from the environment using the actor."""
    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)

    def body(graph, key_):
        action = actor(graph, key_)
        next_graph, reward, cost, done, info = env.step(graph, action)
        return next_graph, (graph, action, reward, cost, done, next_graph)

    keys = jax.random.split(key, env.max_episode_steps)
    final_graph, (graphs, actions, rewards, costs, dones, next_graphs) = \
        jax.lax.scan(body, init_graph, keys, length=env.max_episode_steps)
    data = Rollout(graphs, actions, rewards, costs, dones, next_graphs)
    return data


def rollout_transformer(
        env: MultiAgentEnv,
        policy_apply_fn: Callable,
        policy_params: dict,
        n_agents: int,
        horizon: int,
        key: PRNGKey,
) -> Rollout:
    """Episode rollout using the transformer receding-horizon policy for action selection."""
    from ..algo.receding_horizon import receding_horizon_act

    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)

    def body(graph, key_):
        action, _ = receding_horizon_act(
            policy_apply_fn=policy_apply_fn,
            policy_params=policy_params,
            graph=graph,
            env=env,
            n_agents=n_agents,
            horizon=horizon,
            deterministic=True,
            return_full_rollout=False,
        )

        next_graph, reward, cost, done, info = env.step(graph, action)

        return next_graph, (graph, action, reward, cost, done, next_graph)

    keys = jax.random.split(key, env.max_episode_steps)
    final_graph, (graphs, actions, rewards, costs, dones, next_graphs) = \
        jax.lax.scan(body, init_graph, keys, length=env.max_episode_steps)
    return Rollout(graphs, actions, rewards, costs, dones, next_graphs)


def rollout_dyn_head(
        env: MultiAgentEnv,
        policy_apply_fn: Callable,
        policy_params: dict,
        n_agents: int,
        key: PRNGKey,
) -> tuple:
    """Episode rollout using the dynamics-head world model (no real env.step)."""
    key_x0, _ = jax.random.split(key)
    graph = env.reset(key_x0)

    x         = graph.type_states(type_idx=0, n_type=n_agents)
    goals     = graph.env_states.goal
    obstacles = graph.env_states.obstacle
    car_r     = env._params["car_radius"]
    action_lb, action_ub = env.action_lim()

    obs0        = env._get_obs(x, goals, obstacles)
    obs_history = jnp.repeat(obs0[None], env.history_len, axis=0)
    act_history = jnp.zeros((env.history_len - 1, n_agents, env.action_dim))

    def body(carry, _):
        x_k, obs_hist, act_hist = carry

        obs_seq = obs_hist.transpose(1, 0, 2)
        act_seq = act_hist.transpose(1, 0, 2)
        u_nom   = env.u_nom_from_state(x_k, goals)

        out       = policy_apply_fn(policy_params, obs_seq, act_seq, u_nom, True)
        u_applied = jnp.clip(out.u_applied, action_lb, action_ub)

        x_next   = x_k + out.delta_x_hat
        obs_next = env._get_obs(x_next, goals, obstacles)

        new_obs_hist = jnp.concatenate([obs_hist[1:], obs_next[None]], axis=0)
        new_act_hist = jnp.concatenate([act_hist[1:], u_applied[None]], axis=0)

        is_unsafe = inside_obstacles(x_next[:, :2], obstacles, r=car_r)
        is_finish = jnp.linalg.norm(x_next[:, :2] - goals[:, :2], axis=-1) < car_r * 2

        cost   = is_unsafe.mean()
        reward = -(jnp.linalg.norm(u_applied - u_nom, axis=-1) ** 2).mean()

        return (x_next, new_obs_hist, new_act_hist), (is_unsafe, is_finish, reward, cost)

    _, (step_unsafe, step_finish, step_rewards, step_costs) = jax.lax.scan(
        body,
        (x, obs_history, act_history),
        xs=None,
        length=env.max_episode_steps,
    )
    return step_unsafe, step_finish, step_rewards, step_costs


def has_nan(x):
    """Whether an array has NaNs."""
    return jtu.tree_map(lambda y: jnp.isnan(y).any(), x)


def has_any_nan(x):
    """Whether any array in a pytree has NaNs."""
    return jnp.array(jtu.tree_flatten(has_nan(x))[0]).any()


def compute_norm(grad):
    """Global norm of a gradient pytree."""
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grad)))


def compute_norm_and_clip(grad, max_norm: float):
    """Global gradient norm and clipped gradients."""
    g_norm = compute_norm(grad)
    clipped_g_norm = jnp.maximum(max_norm, g_norm)
    clipped_grad = jtu.tree_map(lambda t: (t / clipped_g_norm) * max_norm, grad)

    return clipped_grad, g_norm


def tree_copy(tree):
    """Copy a pytree."""
    return jtu.tree_map(lambda x: x.copy(), tree)


def empty_grad_tx() -> optax.GradientTransformation:
    """Optax transformation that does nothing."""
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        return None, None

    return optax.GradientTransformation(init_fn, update_fn)


def jax2np(x):
    """Convert a pytree of JAX arrays to NumPy."""
    return jtu.tree_map(lambda y: np.array(y), x)


def np2jax(x):
    """Convert a pytree of NumPy arrays to JAX."""
    return jtu.tree_map(lambda y: jnp.array(y), x)


def is_connected():
    """Whether there is an internet connection."""
    try:
        sock = socket.create_connection(("www.google.com", 80))
        if sock is not None:
            sock.close()
        return True
    except OSError:
        pass
    print('No internet connection')
    return False


def plot_cbf(
        fig: plt.Figure,
        cbf: Callable,
        env: MultiAgentEnv,
        graph: GraphsTuple,
        agent_id: int,
        x_dim: int,
        y_dim: int,
) -> plt.Figure:
    """Plot a CBF contour for one agent."""
    ax = fig.gca()
    n_mesh = 30
    low_lim, high_lim = env.state_lim(graph.states)
    x, y = jnp.meshgrid(
        jnp.linspace(low_lim[x_dim], high_lim[x_dim], n_mesh),
        jnp.linspace(low_lim[y_dim], high_lim[y_dim], n_mesh)
    )
    states = graph.states

    plot_states = states[None, None, :, :].repeat(n_mesh, axis=0).repeat(n_mesh, axis=1)
    plot_states = plot_states.at[:, :, agent_id, x_dim].set(x)
    plot_states = plot_states.at[:, :, agent_id, y_dim].set(y)

    get_new_graph = env.add_edge_feats
    get_new_graph_vmap = jax.vmap(jax.vmap(ft.partial(get_new_graph, graph)))
    new_graph = get_new_graph_vmap(plot_states)
    h = jax.vmap(jax.vmap(cbf))(new_graph)[:, :, agent_id, :].squeeze(-1)
    plt.contourf(x, y, h, cmap=sns.color_palette("rocket", as_cmap=True), levels=15, alpha=0.5)
    plt.colorbar()
    plt.contour(x, y, h, levels=[0.0], colors='blue')
    ax.set_xlim(low_lim[0], high_lim[0])
    ax.set_ylim(low_lim[1], high_lim[1])
    plt.axis('off')

    return fig


def get_bb_cbf(cbf: Callable, env: MultiAgentEnv, graph: GraphsTuple, agent_id: int, x_dim: int, y_dim: int):
    """CBF values on a mesh of positions for one agent."""
    n_mesh = 20
    low_lim = jnp.array([0, 0])
    high_lim = jnp.array([env.area_size, env.area_size])
    b_xs = jnp.linspace(low_lim[x_dim], high_lim[x_dim], n_mesh)
    b_ys = jnp.linspace(low_lim[y_dim], high_lim[y_dim], n_mesh)
    bb_Xs, bb_Ys = jnp.meshgrid(b_xs, b_ys)
    states = graph.states

    bb_plot_states = states[None, None, :, :].repeat(n_mesh, axis=0).repeat(n_mesh, axis=1)
    bb_plot_states = bb_plot_states.at[:, :, agent_id, x_dim].set(bb_Xs)
    bb_plot_states = bb_plot_states.at[:, :, agent_id, y_dim].set(bb_Ys)

    get_new_graph = env.add_edge_feats
    get_new_graph_vmap = jax.vmap(jax.vmap(ft.partial(get_new_graph, graph)))
    bb_new_graph = get_new_graph_vmap(bb_plot_states)
    bb_h = jax.vmap(jax.vmap(cbf))(bb_new_graph)[:, :, agent_id, :].squeeze(-1)
    assert bb_h.shape == (n_mesh, n_mesh)
    return b_xs, b_ys, bb_h


def centered_norm(vmin: float | list[float], vmax: float | list[float]):
    """Color normalization centered at zero."""
    if isinstance(vmin, list):
        vmin = min(vmin)
    if isinstance(vmax, list):
        vmin = max(vmax)
    halfrange = max(abs(vmin), abs(vmax))
    return CenteredNorm(0, halfrange)

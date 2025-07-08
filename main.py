# %% main.py
#   btc2sim main file
# by: Noah Syrkis

# Imports
from functools import partial
from typing import Tuple
import jax.numpy as jnp
from einops import repeat, rearrange
from jax import lax, random, tree, vmap, jit
from jaxtyping import Array
from parabellum.env import Env
from parabellum.types import Config, Obs, State, Action
import parabellum as pb
import nebellum as nb


# %% Constants
c: int = 10  # chunks (how many times to reeval plan)
k: int = 2  # number of imagined futures
m: int = 4  # number of steps into the future to imagine
n: int = 100  # total number of steps in a real sim
s: int = 2  # number of parallel real sims to run
env, cfg = Env(), Config(sims=s, steps=n, knn=5)

# %% Config #####################################################
with open("data/bts.txt", "r") as f:
    bts_str: str = f.read().strip()

with open("data/plan.txt", "r") as f:
    pln_str: str = f.read().strip().split("---")[0].strip()

with open("data/prompt.txt", "r") as f:
    llm_str: str = f.read().strip()


# %% Constants
rng, key = random.split(random.PRNGKey(111))

points = jnp.array([[20, 5], [10, 60]])  # random.randint(rng, (3, 2), 0, cfg.size)
targets = random.randint(rng, (cfg.length,), 0, points.shape[0])

bts, gps = nb.dsl.bts_fn(bts_str), vmap(partial(nb.gps.gps_fn, cfg.map))(points)


@jit
def chamfer_distance(A, B):  # compute distance between two point clouds
    dists: Array = jnp.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    min_dist_A_to_B, min_dist_B_to_A = jnp.min(dists, axis=1), jnp.min(dists, axis=0)
    return jnp.sum(min_dist_A_to_B) + jnp.sum(min_dist_B_to_A)


# %% Functions
def step_fn(env: Env, cfg, behavior, carry: Tuple[Obs, State], rng) -> Tuple[Tuple[Obs, State], Tuple[State, Action]]:
    rngs = random.split(rng, cfg.types.size)
    action: Action = vmap(partial(nb.act.action_fn, env, gps))(rngs, carry[0], behavior, targets)
    obs, state = env.step(cfg, rng, carry[1], action)
    return (obs, state), (state, action)


def chunk_fn(env: Env, cfg: Config, carry: Tuple[Obs, State], rng):
    behavior = nb.act.plan_fn(rng, bts, plan, carry[1])  # perhaps only update plan every m steps
    rngs = random.split(rng, cfg.steps // c)
    (obs, state), seq = lax.scan(partial(step_fn, env, cfg, behavior), carry, rngs)
    aux = lambda x: tree.map(lambda leaf: repeat(leaf, f"... -> {k} ..."), x)  # noqa
    init: Tuple[Obs, State] = aux(obs), aux(encode_fn(cfg, rng, state)[1])
    rngs = random.split(rng, (m, k))
    sim_seq: Tuple[State, Action] = lax.scan(vmap(partial(step_fn, env, cfg, behavior)), init, rngs)[1]
    return (obs, state), (seq, sim_seq)


def encode_fn(cfg: Config, rng, state: State) -> Tuple[str, State, Array]:
    mask = random.bernoulli(rng, 0.0, shape=(cfg.length,))
    mean = jnp.where(~mask[:, None], state.pos, 0).sum(0) / (~mask).sum()
    pos = jnp.where(mask[:, None], mean, state.pos)
    hp = jnp.where(mask, jnp.where(~mask, state.hp, 0).sum() / (~mask).sum(), state.hp)
    return "", State(pos=pos, hp=hp), mask


# @checkify.checkify
def traj_fn(env: Env, cfg: Config, obs: Obs, state: State, rng: Array):
    key, rng = random.split(random.PRNGKey(0))
    obs, state = env.init(cfg, key)
    carry, (seq, sim_seq) = lax.scan(partial(chunk_fn, env, cfg), (obs, state), random.split(rng, c))
    return carry, (seq, sim_seq)  # , action, sim_seq, sim_action)


def plot_fn(seq, sim_seq):
    # c: int = 10  # chunks (how many times to reeval plan)
    # k: int = 2  # number of imagined futures
    # m: int = 4  # number of steps into the future to imagine
    # n: int = 100  # total number of steps in a real sim
    # s: int = 2  # number of parallel real sims to run
    print(tree.map(jnp.shape, sim_seq))
    # seq = tree.map(lambda x: rearrange(x, "s c m k ... -> s (c m k) ...")[:1], seq)
    sim_seq = tree.map(lambda x: rearrange(x, "s c m k ... -> s (c m k) ...")[:2], sim_seq)
    # pb.utils.svg_fn(
    # cfg, seq[0], seq[1], fname="/Users/nobr/desk/s3/nebellum/sims.svg", fps=4, debug=False, targets=points
    # )
    pb.utils.svg_fn(
        cfg, sim_seq[0], sim_seq[1], fname="/Users/nobr/desk/s3/nebellum/sims_sim.svg", fps=4, targets=points
    )


# %%
key_init, rng_traj = random.split(rng, (2, cfg.sims))
plan: nb.types.Plan = tree.map(lambda *x: jnp.stack(x), *tuple(map(partial(nb.lxm.str_to_plan, pln_str, cfg), (-1, 1))))  # type: ignore
obs, state = vmap(partial(env.init, cfg))(key_init)
(obs, state), (seq, sim_seq) = vmap(partial(traj_fn, env, cfg))(obs, state, rng_traj)
plot_fn(seq, sim_seq)

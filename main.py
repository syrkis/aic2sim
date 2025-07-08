# %% main.py
#   btc2sim main file
# by: Noah Syrkis

# Imports
from functools import partial
from typing import Tuple
import jax.numpy as jnp
from einops import repeat
from jax import lax, random, tree, vmap, jit
from jaxtyping import Array
from parabellum.env import Env
from parabellum.types import Config, Obs, State, Action
import nebellum as nb


# %% Constants
c: int = 9
k: int = 11
m: int = 20

# %% Config #####################################################
with open("data/bts.txt", "r") as f:
    bts_str: str = f.read().strip()

with open("data/plan.txt", "r") as f:
    pln_str: str = f.read().strip().split("---")[0].strip()

with open("data/prompt.txt", "r") as f:
    llm_str: str = f.read().strip()


# %% Constants
rng, key = random.split(random.PRNGKey(111))
env, cfg = Env(), Config(sims=3, steps=40, knn=5)

points = jnp.array([[20, 5], [10, 60]])  # random.randint(rng, (3, 2), 0, cfg.size)
targets = random.randint(rng, (cfg.length,), 0, points.shape[0])

bts, gps = nb.dsl.bts_fn(bts_str), vmap(partial(nb.gps.gps_fn, cfg.map))(points)


@jit
def chamfer_distance(A, B):
    dists = jnp.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    min_dist_A_to_B, min_dist_B_to_A = jnp.min(dists, axis=1), jnp.min(dists, axis=0)
    return jnp.sum(min_dist_A_to_B) + jnp.sum(min_dist_B_to_A)


# %% Functions
def step_fn(env, cfg, behavior, carry: Tuple[Obs, State], rng) -> Tuple[Tuple[Obs, State], Tuple[State, Action]]:
    rngs = random.split(rng, cfg.types.size)
    action = vmap(partial(nb.act.action_fn, env, gps))(rngs, carry[0], behavior, targets)
    obs, state = env.step(cfg, rng, carry[1], action)
    return (obs, state), (state, action)


def chunk_fn(env: Env, cfg: Config, carry: Tuple[Obs, State], rng) -> Tuple[Tuple[Obs, State], Tuple[State, State]]:
    behavior = nb.act.plan_fn(rng, bts, plan, carry[1])  # perhaps only update plan every m steps
    rngs = random.split(rng, cfg.steps // c)
    (obs, state), (seq, action) = lax.scan(partial(step_fn, env, cfg, behavior), carry, rngs)
    aux = lambda x: tree.map(lambda leaf: repeat(leaf, f"... -> {k} ..."), x)  # noqa
    init: Tuple[Obs, State] = aux(obs), aux(encode_fn(cfg, rng, state)[1])
    sim_seq: State = lax.scan(vmap(partial(step_fn, env, cfg, behavior)), init, random.split(rng, (m, k)))[1][0]
    return (obs, state), (seq, sim_seq)


def encode_fn(cfg: Config, rng, state: State) -> Tuple[str, State, Array]:
    mask = random.bernoulli(rng, 0.5, shape=(cfg.length,))
    mean = jnp.where(~mask[:, None], state.pos, 0).sum(0) / (~mask).sum()
    pos = jnp.where(mask[:, None], mean, state.pos)
    hp = jnp.where(mask, jnp.where(~mask, state.hp, 0).sum() / (~mask).sum(), state.hp)
    return "", State(pos=pos, hp=hp), mask


# @checkify.checkify
def traj_fn(env: Env, cfg: Config, obs: Obs, state: State, rng: Array) -> Tuple[Tuple[Obs, State], Tuple[State, State]]:
    key, rng = random.split(random.PRNGKey(0))
    obs, state = env.init(cfg, key)
    (obs, state), (seq, sim_seq) = lax.scan(partial(chunk_fn, env, cfg), (obs, state), random.split(rng, c))
    return (obs, state), (seq, sim_seq)


# %%
key_init, rng_traj = random.split(rng, (2, cfg.sims))
plan = tree.map(lambda *x: jnp.stack(x), *tuple(map(partial(nb.lxm.str_to_plan, pln_str, cfg), (-1, 1))))  # type: ignore
obs, state = vmap(partial(env.init, cfg))(key_init)
(obs, state), (seq, sim_seq) = vmap(partial(traj_fn, env, cfg))(obs, state, rng_traj)
print(tree.map(jnp.shape, sim_seq))
# pb.utils.svg_fn(cfg, state_seq, action_seq, "/Users/nobr/desk/s3/aic2sim/sims.svg", fps=4, debug=False, targets=points)

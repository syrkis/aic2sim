# %% main.py
#   btc2sim main file
# by: Noah Syrkis

# Imports
from functools import partial
from typing import Tuple

import esch
import jax.numpy as jnp
import numpy as np
import parabellum as pb
from einops import rearrange, repeat
from gemma import gm
from jax import jit, lax, random, tree, vmap
from jaxtyping import Array
from parabellum.env import Env
from parabellum.types import Action, Config, Obs, State

import nebellum as nb


# %% Constants
c: int = 2  # chunks (how many times to reeval plan)
n: int = 100  # total number of steps in a real sim
k: int = 2  # number of imagined futures
s: int = 2  # number of parallel real sims to run
m: int = n // c  # number of steps into the future to imagine
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
# model = gm.nn.Gemma3_1B()
# tokenizer = gm.text.Gemma3Tokenizer()
# params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)
# sampler = gm.text.Sampler(model=model, params=params)


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
    step = partial(step_fn, env, cfg, behavior)
    aux = lambda x: tree.map(lambda leaf: repeat(leaf, f"... -> {k} ..."), x)  # noqa
    init: Tuple[Obs, State] = aux(carry[0]), aux(decode_fn(*encode_fn(cfg, rng, carry[1])))
    sim_seq: Tuple[State, Action] = lax.scan(vmap(step), init, random.split(rng, (m, k)))[1]
    carry, seq = lax.scan(step, carry, random.split(rng, cfg.steps // c))
    return carry, (seq, sim_seq)


def encode_fn(cfg: Config, rng, state: State) -> Tuple[str, State, Array]:
    mask = random.bernoulli(rng, 0.0, shape=(cfg.length,))
    mean = jnp.where(~mask[:, None], state.pos, 0).sum(0) / (~mask).sum()
    pos = jnp.where(mask[:, None], mean, state.pos)
    hp = jnp.where(mask, jnp.where(~mask, state.hp, 0).sum() / (~mask).sum(), state.hp)
    return "", State(pos=pos, hp=hp), mask

def decode_fn(intel: str, state: State, mask: Array) -> State:
    return state
    ## testing GAMMA INSIDE OF JAX STUFF
    prompt = tokenizer.encode("One word to describe Paris: \n\n", add_bos=True)
    prompt = jnp.asarray(prompt)
    out = model.apply({"params": params}, tokens=prompt, return_last_only=True)
    next_token = random.categorical(random.key(1), out.logits)
    tokenizer.decode(next_token)
    ## GAMMA TEST END


# @checkify.checkify
def traj_fn(env: Env, cfg: Config, obs: Obs, state: State, rng: Array):
    key, rng = random.split(random.PRNGKey(0))
    obs, state = env.init(cfg, key)
    carry, (seq, sim_seq) = lax.scan(partial(chunk_fn, env, cfg), (obs, state), random.split(rng, c))
    return carry, (seq, sim_seq)  # , action, sim_seq, sim_action)


def plot_fn(seq, sim_seq):
    # print(tree.map(jnp.shape, sim_seq))
    # print(tree.map(jnp.shape, seq))
    seq = tree.map(lambda x: rearrange(x[0], "c m ... -> 1 (c m) ..."), seq)
    sim_seq = tree.map(lambda x: rearrange(x[0], "c m k ... -> k (c m) ..."), sim_seq)
    pb.utils.svg_fn(cfg, seq[0], seq[1], fname="/Users/nobr/desk/s3/nebellum/seqs.svg", fps=2, targets=points)
    pb.utils.svg_fn(cfg, sim_seq[0], sim_seq[1], fname="/Users/nobr/desk/s3/nebellum/sims.svg", fps=2, targets=points)
    tmp = np.array(
        vmap(lambda x, y: ((x - y) ** 2).sum())(seq[0].pos[0], sim_seq[0].pos[0])[None, None, ...], dtype=float
    )
    tmp /= tmp.max()
    e = esch.Drawing(w=1 - 1, h=tmp.size - 1, row=1, col=1)
    esch.grid_fn(e, tmp * 0.8, shape="square")
    e.dwg.saveas("/Users/nobr/desk/s3/nebellum/diff.svg")
    # print(rearrange(seq[0].pos, "a b ... -> (a b)"))

    # .shape, sim_seq[1].pos.shape)


# %%
key_init, rng_traj = random.split(rng, (2, cfg.sims))
plan: nb.types.Plan = tree.map(lambda *x: jnp.stack(x), *tuple(map(partial(nb.lxm.str_to_plan, pln_str, cfg), (-1, 1))))  # type: ignore
obs, state = vmap(partial(env.init, cfg))(key_init)
(obs, state), (seq, sim_seq) = vmap(partial(traj_fn, env, cfg))(obs, state, rng_traj)
plot_fn(seq, sim_seq)

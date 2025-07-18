# %% main.py
#   btc2sim main file
# by: Noah Syrkis

# Imports
from dataclasses import replace
from functools import partial
from typing import Tuple

import esch
import jax.numpy as jnp
import numpy as np
import parabellum as pb
from einops import rearrange, repeat
from jax import jit, lax, random, tree, vmap
from jaxtyping import Array
from parabellum.env import Env
from parabellum.types import Action, Config, Obs, State
from flax.traverse_util import flatten_dict

import nebellum as nb

#  model

# %% Constants
c: int = 4  # chunks (how many times to reeval plan)
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

with open("data/intel.txt", "r") as f:
    intel_str: str = f.read().strip()

# %% Constants
points = jnp.array([[20, 5], [10, 60]])  # random.randint(rng, (3, 2), 0, cfg.size)
targets = random.randint(random.PRNGKey(0), (cfg.length,), 0, points.shape[0])
bts, gps = nb.dsl.bts_fn(bts_str), vmap(partial(nb.gps.gps_fn, cfg.map))(points)
model, params, tokenizer = nb.llm.load_fn()
# flat_params = flatten_dict(params)
# for k in flat_params:
# print(k)

# exit()
prompt = jnp.asarray(tokenizer.encode("It was the worst of times, it was the best of times.", add_bos=True))
out = model.apply({"params": params}, tokens=prompt, return_last_only=True)
# print(params["layer_13"]["mlp"].keys())
exit()
template = jnp.array([jnp.pad(jnp.array(t), (42 - len(t), 0)) for t in map(tokenizer.encode, intel_str.split("\n"))])


print(out)
exit()


# print([tokenizer.encode(str(i)) for i in range(100)])
# print(tokenizer.encode(" "))
# print(tokenizer.encode(""))
# exit()


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


def chunk_fn(
    env: Env, cfg: Config, carry: Tuple[Obs, State], rng
) -> Tuple[Tuple[Obs, State], Tuple[Tuple[State, Action], Tuple[State, Action], Array]]:
    behavior = nb.act.plan_fn(rng, bts, plan, carry[1])  # perhaps only update plan every m steps
    step = partial(step_fn, env, cfg, behavior)
    init, mask = distort_fn(carry, cfg, rng)
    sim_seq: Tuple[State, Action] = lax.scan(vmap(step), init, random.split(rng, (m, k)))[1]
    carry, seq = lax.scan(step, carry, random.split(rng, cfg.steps // c))
    return carry, (seq, sim_seq, mask)


def distort_fn(carry, cfg, rng) -> Tuple[Tuple[Obs, State], Array]:
    aux = lambda x: tree.map(lambda leaf: repeat(leaf, f"... -> {k} ..."), x)  # noqa (copy state and obs)
    obs, state = aux(carry[0]), aux(decode_fn(*encode_fn(rng, carry[1])))  # type: ignore
    mask = random.bernoulli(rng, 0.5, shape=(1, cfg.length)) & random.bernoulli(rng, 0.7, shape=(k, cfg.length))
    state = replace(state, pos=jnp.where(mask[..., None], state.pos, state.pos.mean()))  # * mask[..., None])
    return (obs, state), mask


def encode_fn(rng, state: State) -> Tuple[Array, State, Array]:
    # what units to hide
    mask = (random.permutation(rng, state.pos.shape[0]) < (state.pos.shape[0] // 2)).astype(jnp.bool_)

    # mean position to assign to hidden units: TODO: introduce NONE values
    pos = jnp.where(mask[:, None], jnp.where(~mask[:, None], state.pos, 0).sum(0) / (~mask).sum(), state.pos)

    # Same for hp
    hp = jnp.where(mask, jnp.where(~mask, state.hp, 0).sum() / (~mask).sum(), state.hp)

    # make an intel array for gemma
    intel: Array = template[random.randint(rng, (state.pos.shape[0] // 2,), minval=0, maxval=template.shape[0])]

    # debug.print("{x}", x=intel)
    # debug.print("{x}", x=lax.map(lambda x:jnp.where(intel[0] == 6, 99, intel))

    return intel, State(pos=pos, hp=hp), mask


def loc_to_tok(x, y):
    pass


def decode_fn(intel: Array, state: State, mask: Array) -> State:
    return state
    ## testing GAMMA INSIDE OF JAX STUFl
    # prompt = tokenizer.encode("One word to describe Paris: \n\n", add_bos=True)
    # prompt = jnp.asarray(prompt)
    # out = model.apply({"params": params}, tokens=prompt, return_last_only=True)
    # next_token = random.categorical(random.key(1), out.logits)
    # tokenizer.decode(next_token)
    ## GAMMA TEST END


# @checkify.checkify
def traj_fn(env: Env, cfg: Config, rng: Array):
    key, rng = random.split(random.PRNGKey(0))
    obs, state = env.init(cfg, key)
    rngs = random.split(rng, c)
    carry, (seq, sim_seq, mask) = lax.scan(partial(chunk_fn, env, cfg), (obs, state), rngs)
    return carry, (seq, sim_seq, mask)  # , action, sim_seq, sim_action)


def plot_fn(seq, sim_seq, mask):
    # print(tree.map(jnp.shape, sim_seq))
    # print(tree.map(jnp.shape, seq))
    seq = tree.map(lambda x: rearrange(x[0], "c m ... -> 1 (c m) ..."), seq)
    new_sim_seq = tree.map(lambda x: rearrange(x[0], "c m k ... -> k (c m) ..."), sim_seq)
    pb.utils.svg_fn(cfg, seq[0], seq[1], fname="/Users/nobr/desk/s3/nebellum/seqs.svg", fps=2, targets=points)
    pb.utils.svg_fn(
        cfg, new_sim_seq[0], new_sim_seq[1], fname="/Users/nobr/desk/s3/nebellum/sims.svg", fps=2, targets=points
    )
    tmp = np.array(
        vmap(lambda x, y: ((x - y) ** 2).sum())(seq[0].pos[0], new_sim_seq[0].pos[0])[None, None, ...], dtype=float
    )

    tmp /= tmp.max()

    e = esch.Drawing(w=1 - 1, h=tmp.size - 1, row=1, col=1)
    esch.grid_fn(e, tmp * 0.8, shape="square")
    e.dwg.saveas("/Users/nobr/desk/s3/nebellum/diff.svg")

    #  GOOD NIGHT AFTER THIS
    e = esch.Drawing(h=32, w=9, row=1, col=1)
    seq_tmp = np.array(((sim_seq[0].pos[0][0] - seq[0].pos[0][0]) ** 2).sum((-1, -2)), dtype=float).T[None, ...] ** 2
    seq_tmp /= seq_tmp.max()
    esch.grid_fn(e, seq_tmp, shape="square")
    esch.save(e.dwg, "/Users/nobr/desk/s3/nebellum/seq_tmp.svg")

    e = esch.Drawing(h=9, w=5, row=1, col=1)
    tmp = np.array(mask[0][0].T, dtype=float)[None, ...]
    tmp = tmp / tmp.max() * 0.8
    esch.grid_fn(e, tmp, shape="square")
    esch.save(e.dwg, "/Users/nobr/desk/s3/nebellum/mask_tmp.svg")
    # print(rearrange(seq[0].pos, "a b ... -> (a b)"))

    # .shape, sim_seq[1].pos.shape)


# %%
rng = random.PRNGKey(0)
key_init, rng_traj = random.split(rng, (2, cfg.sims))
plan: nb.types.Plan = tree.map(lambda *x: jnp.stack(x), *tuple(map(partial(nb.dsl.str_to_plan, pln_str, cfg), (-1, 1))))  # type: ignore
(obs, state), (seq, sim_seq, mask) = vmap(partial(traj_fn, env, cfg))(rng_traj)
# plot_fn(seq, sim_seq, mask)

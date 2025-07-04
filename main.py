# %% main.py
#   btc2sim main file
# by: Noah Syrkis

# Imports
from functools import partial
from typing import Tuple

from jax.experimental import checkify
import jax.numpy as jnp
from jax import lax, random, tree, vmap
from jaxtyping import Array
import parabellum as pb
from parabellum.env import Env
from parabellum.types import Config, Obs, State

import aic2sim as a2s

# %% Config #####################################################
with open("data/bts.txt", "r") as f:
    bts_str: str = f.read().strip()

with open("data/plan.txt", "r") as f:
    pln_str: str = f.read().strip().split("---")[0].strip()

with open("data/prompt.txt", "r") as f:
    llm_str: str = f.read().strip()


# %% Constants
rng, key = random.split(random.PRNGKey(111))
env, cfg = Env(), Config(sims=1, steps=40)

# points = jnp.int32(random.uniform(rng, (1, 2), minval=0, maxval=cfg.size))
points = jnp.array([[20, 5]])
targets = random.randint(rng, (cfg.length,), 0, points.shape[0])

bts, gps = a2s.dsl.bts_fn(bts_str), vmap(partial(a2s.gps.gps_fn, cfg.map))(points)
action_fn = vmap(partial(a2s.act.action_fn, env, gps))


# %% Functions
def step_fn(env: Env, cfg: Config, carry: Tuple[Obs, State], rng: Array):
    obs, state = carry
    rngs = random.split(rng, cfg.length)
    behavior = a2s.act.plan_fn(rng, bts, plan, state)  # perhaps only update plan every m steps
    action = action_fn(rngs, obs, behavior, targets)
    obs, state = env.step(cfg, rng, state, action)
    checkify.check(~action.invalid, "Action is not valid")  # MUST return a valid action
    return (obs, state), (state, action)


# rin a single simulation
@checkify.checkify
def traj_fn(env, cfg, obs, state, rng):
    rngs = random.split(rng, cfg.steps)
    return lax.scan(partial(step_fn, env, cfg), (obs, state), rngs)


# %%
key_init, rng_traj = random.split(rng, (2, cfg.sims))
plan = tree.map(lambda *x: jnp.stack(x), *tuple(map(partial(a2s.lxm.str_to_plan, pln_str, cfg), (-1, 1))))  # type: ignore
obs, state = vmap(partial(env.init, cfg))(key_init)
err, ((obs, state), (state_seq, action_seq)) = vmap(partial(traj_fn, env, cfg))(obs, state, rng_traj)
# err.throw()
pb.utils.svg_fn(cfg, state_seq, action_seq, "/Users/nobr/desk/s3/aic2sim/sims.svg", fps=10, debug=True, targets=points)

# act.py
#   returns an action function that applies a plan
# by: Noah Syrkis


# imports
import jax.numpy as jnp
from jax.experimental import checkify
from jaxtyping import Array
from parabellum.env import Env
from dataclasses import replace
from parabellum.types import Obs
from typing import Tuple, List, Callable
import equinox as eqx
from jax import lax, tree, random, debug
from parabellum.types import Action, State
from aic2sim.types import Tree, Leaf, Compass, Plan, FAILURE, SUCCESS, INITIAL


# %% Globals
MOVE = jnp.array(1)
CAST = jnp.array(2)
STAY, NONE = Action(kind=jnp.array(1), pos=jnp.zeros(2)), Action(kind=jnp.array(0), pos=jnp.zeros(2))


# %% Tree Treefunctions
def plan_fn(rng: Array, bts, plan: Plan, state: State) -> Tree:  # TODO: Focus
    def move(step):  # all units in focus within 10 meters of target position (fix quadratic)
        return ((jnp.linalg.norm(state.pos - step.coord) * step.units) < 10).all()

    def kill(step):  # all enemies dead within 10 meters of target  (this is quadratric and should be made smart)
        return ((jnp.linalg.norm(state.pos - step.coord) * ~step.units * (state.hp == 0)) < 10).any()

    def aux(plan: Plan):
        cond = lax.map(lambda step: lax.cond(step.move, move, kill, step), plan)
        # debug.breakpoint()
        # process cond better than argmin by scanning, through children.
        # idx = scan and mask through children (use instead of cond.argmin())
        return plan.btidx[cond.argmin()] * plan.units[cond.argmin()]

    idxs = lax.map(aux, plan).sum(0)  # mapping across teams (2 for now, but supports any number)
    return tree.map(lambda x: jnp.take(x, idxs, axis=0), bts)  # behavior


# @eqx.filter_jit
def fmap(fns, rng: Array, obs: Obs, gps: Compass, target: Array, bt: Tree) -> Leaf:
    rngs = random.split(rng, len(fns))
    leafs = [f(rng, obs, gps, target) for f, rng in zip(fns, rngs)]
    select = lambda *xs: jnp.stack(xs).take(bt.idxs, axis=0)  # noqa # .take() is reodering the bt to leaf order
    return Leaf(status=tree.map(select, *leafs.status), action=tree.map(select, *leafs.action))  # type: ignore


def action_fn(env: Env, gps: Compass, rng: Array, obs: Obs, bt: Tree, target: Array) -> Action:
    rngs = random.split(rng, len(fns))
    calls = tuple((f(rng, obs, gps, target) for f, rng in zip(fns, rngs)))
    status = jnp.stack([c.status for c in calls]).take(bt.idxs)
    action = tree.map(lambda *x: jnp.stack(x).take(bt.idxs, axis=0), *[c.action for c in calls])  # type: ignore
    leafs = Leaf(action=action, status=status, jump=jnp.zeros(len(fns)))  # jump will not be used
    init = Leaf(action=NONE, status=INITIAL, jump=jnp.array(0))
    state, flag = lax.scan(bt_fn, init, (leafs, bt))
    # checkify.check(~leaf.invalid, "Action is not valid")  # MUST return a valid action
    return state.action


def bt_fn(state: Leaf, input: Tuple[Leaf, Tree]) -> Tuple[Leaf, Array]:  # TODO: account for cond versus action leaf
    leaf, node = input  # load atomics and bt status

    look = (state.failure | state.initial) & ~state.jump  # should we even look?

    flag = look & ((node.sequence & ~state.failure) | (node.fallback & ~state.success))  # if we look, should we use?

    status = jnp.where(flag, leaf.status, state.status)  # update status if we should

    action: Action = lax.cond(flag, lambda: leaf.action, lambda: state.action)  # update action if we should

    leaf = Leaf(action=action, status=status)  # make new leaf

    jump = jnp.where((node.over & leaf.success) | (~node.over & leaf.failure), state.jump - 1, node.jump)  # jumps?

    return lax.cond(flag, lambda: replace(leaf, jump=jump), lambda: leaf), flag  # return


###################################################################################
# %% Actions ######################################################################
###################################################################################
def stand_fn(rng: Array, obs: Obs, gps: Compass, target: Array) -> Leaf:
    return Leaf(action=STAY, status=jnp.array(0))


def move_fn(rng: Array, obs: Obs, gps: Compass, target: Array) -> Leaf:
    pos = jnp.int32(obs.pos[0])
    pos = -jnp.array((gps.dy[target][*pos], gps.dx[target][*pos])) * obs.speed[0]
    action = Action(pos=pos, kind=MOVE)
    return Leaf(action=action, status=SUCCESS)


def attack_fn(rng: Array, obs: Obs, gps: Compass, target: Array) -> Leaf:
    idx = random.choice(rng, a=jnp.arange(obs.enemy.size), p=obs.enemy)
    action = Action(pos=obs.pos[idx], kind=CAST)
    status = lax.cond(idx != 0, lambda: 0, lambda: 1)
    return Leaf(action=action, status=status)


###################################################################################
# %% Conditions ###################################################################
###################################################################################
def enemy_in_reach_fn(rng: Array, obs: Obs, gps: Compass, target: Array) -> Leaf:
    status = (obs.enemy * (obs.dist < obs.reach[0])).sum() > 0
    return Leaf(status=status, action=NONE)


# def alive_fn(rng: Array, obs: Obs, gps: Compass, target: Array):
# status = Status(status=(obs.hp[0] > 0))
# return status, NONE


# def enemy_in_sight_fn(rng: Array, obs: Obs, gps: Compass, target: Array):
# debug.breakpoint()
# return Status(status=obs.enemy.any() > 0), NONE


# def ally_in_sight_fn(rng: Array, obs: Obs, gps: Compass, target: Array):
# return Status(status=obs.ally.any() > 0), NONE
#
#
# def ally_in_reach_fn(rng: Array, obs: Obs, gps: Compass, target: Array):
# status = Status(status=(obs.ally * obs.dist < obs.reach[0]).sum() > 0)
# return status, NONE


###################################################################################
# %% Grammar ######################################################################
###################################################################################
tuples = sorted(
    [
        (("in_reach", "enemy"), enemy_in_reach_fn),
        (("move", "target"), move_fn),
        (("shoot", "random"), attack_fn),
        # (("in_reach", "ally"), ally_in_reach_fn),
        # (("stand",), stand_fn),
        # (("in_sight", "enemy"), enemy_in_sight_fn),
        # (("is_alive",), alive_fn),
        # (("in_sight", "ally"), ally_in_sight_fn),
        # (("shoot", "closest"), shoot_closest_fn),
    ],
    key=lambda x: x[0],
)
a2i = {a[0]: idx for idx, a in enumerate(tuples)}
fns = tuple((a[1] for a in tuples))  # type: ignore

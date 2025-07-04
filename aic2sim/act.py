# act.py
#   returns an action function that applies a plan
# by: Noah Syrkis


# imports
import jax.numpy as jnp
from jaxtyping import Array
from parabellum.env import Env
from parabellum.types import Obs
from typing import Tuple
from jax import lax, tree, random, debug
from parabellum.types import Action, State
from aic2sim.types import Tree, Leaf, Compass, Plan


# %% Auxilary
def action_aux(action) -> Leaf:
    return Leaf(action=action, status=jnp.array(True), cond=jnp.array(False), jump=jnp.array(0))


def cond_aux(status) -> Leaf:
    action = Action(move=jnp.array(True), pos=jnp.zeros(2))
    return Leaf(status=status, cond=jnp.array(False), jump=jnp.array(0), action=action)


# %% Tree functions
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


def action_fn(env: Env, gps: Compass, rng: Array, obs: Obs, bt: Tree, target: Array) -> Action:
    rngs = random.split(rng, len(fns))
    calls: Tuple[Leaf, ...] = tuple((f(rng, obs, gps, target) for f, rng in zip(fns, rngs)))

    status, cond = jnp.stack([c.status for c in calls]).take(bt.idxs), jnp.stack([c.cond for c in calls]).take(bt.idxs)
    action: Action = tree.map(lambda *x: jnp.stack(x).take(bt.idxs, axis=0), *[c.action for c in calls])  # type: ignore

    init = cond_aux(jnp.array(False))  # Leaf(action=NONE, status=jnp.array(False),
    leafs = Leaf(action=action, status=status, jump=jnp.int32(cond * 0), cond=cond)

    state, flag = lax.scan(bt_fn, init, (leafs, bt))
    # debug.print("{i}", i=state.action.cast)
    return state.action


def bt_fn(state: Leaf, input: Tuple[Leaf, Tree]) -> Tuple[Leaf, Array]:  # TODO: account for cond versus action leaf
    leaf, node = input  # load atomics and bt status

    flag = ~jnp.bool(state.jump) & ((node.all & ~state.failure) | (node.one & ~state.success))

    status = jnp.where(flag, leaf.status, state.status)  # update status if we should

    action: Action = lax.cond(flag & ~leaf.cond, lambda: leaf.action, lambda: state.action)  # maybe update action

    jump = jnp.where(state.jump > 0, state.jump - 1, node.jump) * ((node.all & status) | (node.one & ~status))

    leaf = Leaf(action=action, status=status, jump=jump, cond=jnp.array(False))
    return leaf, flag


###################################################################################
# %% Actions ######################################################################
###################################################################################
def stand_fn(rng: Array, obs: Obs, gps: Compass, target: Array) -> Leaf:
    action = Action(pos=jnp.zeros(2, dtype=jnp.int32), move=jnp.array(True))
    return action_aux(action)


def move_fn(rng: Array, obs: Obs, gps: Compass, target: Array) -> Leaf:
    pos = -jnp.array((gps.dy[target][*jnp.int32(obs.pos[0])], gps.dx[target][*jnp.int32(obs.pos[0])])) * obs.speed[0]
    action = Action(pos=pos, move=jnp.array(True))
    return action_aux(action)


def attack_fn(rng: Array, obs: Obs, gps: Compass, target: Array) -> Leaf:  # TODO: why i need to flip pos ::-1?
    idx = random.choice(rng, a=jnp.arange(obs.enemy.size), p=obs.enemy)
    action = Action(pos=jnp.array([1, 1]) * obs.pos[idx][::-1] * (idx != 0), move=jnp.array(False))
    # debug.print("{i}", i=action.pos)
    return action_aux(action)


###################################################################################
# %% Conditions ###################################################################
###################################################################################
def enemy_in_reach_fn(rng: Array, obs: Obs, gps: Compass, target: Array) -> Leaf:
    return cond_aux((obs.enemy * (obs.dist < obs.reach[0])).sum() > 0)


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

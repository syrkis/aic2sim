# act.py
#   returns an action function that applies a plan
# by: Noah Syrkis


# imports
import jax.numpy as jnp
from jax.experimental import checkify
from jaxtyping import Array
from parabellum.env import Env
from parabellum.types import Obs
from typing import Tuple
import equinox as eqx
from jax import lax, tree, random, debug
from parabellum.types import Action, State
from aic2sim.types import Behavior, Status, Compass, Plan


# %% Globals
CAST = jnp.array(2)
MOVE = jnp.array(1)

STAND = Action(kind=jnp.array(1), pos=jnp.zeros(2))
NONE = Action(kind=jnp.array(0), pos=jnp.zeros(2))

SUCCESS = Status(status=jnp.array(True))
FAILURE = Status(status=jnp.array(False))


# %% Behavior Treefunctions
def plan_fn(rng: Array, bts, plan: Plan, state: State) -> Behavior:  # TODO: Focus
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


@eqx.filter_jit
def fmap(fns, rng: Array, obs: Obs, gps: Compass, target: Array, bt: Behavior):
    status, action = tuple(zip(*(f(rng, obs, gps, target) for f, rng in zip(fns, random.split(rng, len(fns))))))
    select = lambda *xs: jnp.stack(xs).take(bt.idx, axis=0)  # noqa # .take() is reodering the bt to leaf order
    return tree.map(select, *status), tree.map(select, *action)  # type: ignore


def action_fn(env: Env, gps: Compass, rng, obs: Obs, bt: Behavior, target: Array) -> Action:
    atom_status, atom_action = fmap(fns, rng, obs, gps, target, bt)
    init = (Status(), NONE, jnp.array(0))
    xs = atom_status, atom_action, bt, jnp.arange(len(fns))
    debug.print("{team}", team=obs.team[0])
    (_, action, _), flag = lax.scan(bt_fn, init, xs)
    checkify.check(~action.invalid, "Action is not valid")  # MUST return a valid action
    debug.print("\n\n——————\n\n")
    return action


def bt_fn(carry: Tuple[Status, Action, Array], input: Tuple[Status, Action, Behavior, Array]):  # this is wrong
    debug.print("team: {team} \t\t {input}", team=0, input=input[-1])
    atom_status, atom_action, bt, idx = input  # load atomics and bt status
    prev_status, prev_action, passing = carry

    search = prev_status.failure | prev_action.invalid  # almost certainly right
    checks = (bt.prev & ~prev_status.failure) | (~bt.prev & ~prev_status.success) | (idx == 0)  # probably right

    status = Status(status=jnp.where(search & checks & (passing <= 0), atom_status.status, prev_status.status))
    action = tree.map(lambda x, y: jnp.where(search & checks & (passing <= 0), x, y), atom_action, prev_action)

    flag = (bt.parent & status.failure) | (~bt.parent & status.success)  # update passing
    passing = jnp.where(search & checks & (passing <= 0), jnp.where(flag, passing - 1, bt.skip), passing)
    return (status, action, passing), flag


###################################################################################
# %% Actions ######################################################################
###################################################################################
def stand_fn(rng: Array, obs: Obs, gps: Compass, target: Array):
    return SUCCESS, STAND


def move_fn(rng: Array, obs: Obs, gps: Compass, target: Array):
    pos = jnp.int32(obs.pos[0])
    pos = -jnp.array((gps.dy[target][*pos], gps.dx[target][*pos])) * obs.speed[0]
    action = Action(pos=pos, kind=MOVE)
    # debug.print("move\tteam: {team}\t\taction: {action}", action=action.pos.round(), team=obs.team[0])
    return SUCCESS, action


def attack_fn(rng: Array, obs: Obs, gps: Compass, target: Array):
    idx = random.choice(rng, a=jnp.arange(obs.enemy.size), p=obs.enemy)
    action = Action(pos=obs.pos[idx], kind=CAST)
    status = lax.cond(idx != 0, lambda: SUCCESS, lambda: FAILURE)
    # debug.print("{status} {action}", status=status.status, action=action.pos.round().flatten())
    return status, action


###################################################################################
# %% Conditions ###################################################################
###################################################################################
def enemy_in_reach_fn(rng: Array, obs: Obs, gps: Compass, target: Array):
    status = Status(status=(obs.enemy * (obs.dist < obs.reach[0])).sum() > 0)
    # debug.print("reach\tteam: {team} \tpos: {pos}", team=obs.team[0], pos=obs.pos.flatten().round())
    return status, NONE


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
        (("move", "target"), move_fn),
        (("shoot", "random"), attack_fn),
        (("in_reach", "enemy"), enemy_in_reach_fn),
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
fns = tuple((a[1] for a in tuples))

# imports
from chex import dataclass
from dataclasses import dataclass as _dataclass
from jaxtyping import Array, Bool
from typing import List
import jax.numpy as jnp
import parabellum as pb
from dataclasses import field
from parabellum.types import Config, Action
from parabellum.env import Env


SUCCESS = jnp.array(True)
FAILURE = jnp.array(False)


# dataclasses
@dataclass
class Plan:
    units: Array  # Bool  # one hot of what units are in
    coord: Array
    btidx: Array
    parent: Array  #
    move: Array  # or kill


@dataclass
class Leaf:
    action: Action
    status: Bool[Array, "..."]
    cond: Bool[Array, "..."]
    jump: Array

    @property
    def success(self) -> Bool[Array, "..."]:
        return self.status

    @property
    def failure(self) -> Bool[Array, "..."]:
        return ~self.status


@dataclass
class Tree:  # there will be one per unit (called wihth differnt obs)
    idxs: Array
    over: Array
    left: Array
    jump: Array

    @property
    def one(self) -> Array:
        return ~self.left

    @property
    def all(self) -> Array:
        return self.left

    def __repr__(self):
        return f"""
idxs: {self.idxs}
over: {self.over}
left: {self.left}
jump: {self.jump}
"""


@dataclass
class Compass:  # groups can have targets
    point: Array
    df: Array
    dy: Array
    dx: Array


@dataclass
class Battalion:
    units: Array  # bool array in batalion else 0
    target: Array  # 0 to 6
    bt_idx: Array  # 0 to num bts


# %% Types
@_dataclass
class Step:
    rng: Array
    obs: pb.types.Obs
    state: pb.types.State
    action: pb.types.Action | None


@_dataclass
class Game:
    rng: List[Array]
    env: Env
    cfg: Config
    # step_fn: pb.env.step_fn
    gps: Compass
    step_seq: List[Step]
    messages: List[dict]

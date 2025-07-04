# imports
from chex import dataclass
from dataclasses import dataclass as _dataclass
from jaxtyping import Array
from typing import List
import jax.numpy as jnp
import parabellum as pb
from dataclasses import field
from parabellum.types import Config, Action
from parabellum.env import Env


SUCCESS = jnp.array(True)
FAILURE = jnp.array(False)
INITIAL = jnp.array(-1)


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
    status: Array
    jump: Array = field(default_factory=lambda: jnp.array(0))

    @property
    def success(self) -> Array:
        return self.status == SUCCESS

    @property
    def failure(self) -> Array:
        return self.status == FAILURE

    @property
    def initial(self) -> Array:
        return self.status == INITIAL

    @property
    def condition(self) -> Array:
        return self.action.invalid


@dataclass
class Tree:  # there will be one per unit (called wihth differnt obs)
    idxs: Array
    over: Array
    left: Array
    jump: Array

    @property
    def fallback(self) -> Array:
        return ~self.left

    @property
    def sequence(self) -> Array:
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

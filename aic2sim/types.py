# imports
from chex import dataclass
from dataclasses import dataclass as _dataclass
from dataclasses import field
from jaxtyping import Array
from typing import List
import jax.numpy as jnp
import parabellum as pb
from parabellum.types import Config
from parabellum.env import Env


# dataclasses
@dataclass
class Plan:
    units: Array  # Bool  # one hot of what units are in
    coord: Array
    btidx: Array
    parent: Array  #
    move: Array  # or kill


@dataclass
class Status:
    status: Array = field(default_factory=lambda: jnp.array(True))  # status
    active: Array = field(default_factory=lambda: jnp.array(False))  # active

    @property
    def success(self):
        return self.status

    @property
    def failure(self):
        return ~self.status


@dataclass
class Behavior:  # there will be one per unit (called wihth differnt obs)
    idx: Array
    parent: Array
    prev: Array
    skip: Array


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

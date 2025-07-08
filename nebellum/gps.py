# gps.py
#   calculates a navigation path from everywhere on terrain to target
# by: Noah Syrkis


# Imports
import jax.numpy as jnp
import jax.lax as lax
from nebellum.types import Compass


kernel = jnp.array([[jnp.sqrt(2), 1, jnp.sqrt(2)], [1, 0, 1], [jnp.sqrt(2), 1, jnp.sqrt(2)]]).reshape((1, 1, 3, 3))


def gps_fn(map, point) -> Compass:
    def step_fn(carry, step):
        front, df = carry
        front = (lax.conv(front, kernel, (1, 1), "SAME") > 0) * (df[None, None, ...] == front.size) * mask
        df = jnp.where(front, step, df).squeeze()  # type: ignore
        return (front, df), None

    mask = jnp.float32(jnp.abs(map - 1))
    front = jnp.zeros(map.shape).at[*point].set(1)[None, None, ...]
    df = jnp.where(front, 0, front.size).squeeze()
    steps = jnp.arange(map.shape[0] * 2)
    front, df = lax.scan(step_fn, (front, df), steps)[0]
    dy, dx = jnp.gradient(df * ~(point == 0).all())  # 0,0 is an INVLAID target
    return Compass(point=point, df=df, dy=dy, dx=dx)

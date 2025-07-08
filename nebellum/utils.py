# imports
import jax.numpy as jnp

# %% Dicts
nato_to_int = dict(alpha=0, bravo=1, charlie=2, delta=3, echo=4, foxtrot=5)
int_to_nato = {v: k for k, v in nato_to_int.items()}

chess_to_int = dict(pawn=0, rook=1, knight=2, bishop=3, queen=4, king=5)
int_to_chess = {v: k for k, v in chess_to_int.items()}

bt_to_int = dict(scout=0, patrol=1, assault=2, defend=3, flank=4, retreat=5, ambush=6, support=7, recon=8, siege=9)
int_to_bt = {v: k for k, v in bt_to_int.items()}

alpha_to_int = dict(A=0, B=1, C=2, D=3, E=4, F=5)
int_to_alpha = {v: k for k, v in alpha_to_int.items()}


# %% Functions
def scene_fn(arr):
    arr = jnp.zeros_like(arr)
    start_idx = arr.shape[0] // 6
    end_idx = arr.shape[0] // 6 * 2
    arr = arr.at[start_idx + 30 : end_idx + 30, start_idx:end_idx].set(1)
    arr = arr.at[start_idx + 30 + 5 : end_idx + 30 - 5, start_idx + 5 : end_idx].set(0)
    return arr


def typst_fn(bt_str):
    pass

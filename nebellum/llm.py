import os
from gemma import gm, peft
from flax import serialization
import jax.numpy as jnp


def rename_w_to_kernel(params):
    # Recursively rename "w" keys to "kernel"
    if isinstance(params, dict):
        new_dict = {}
        for k, v in params.items():
            if k == "w":
                new_dict["kernel"] = rename_w_to_kernel(v)
            else:
                new_dict[k] = rename_w_to_kernel(v)
        return new_dict
    elif isinstance(params, list):
        return [rename_w_to_kernel(x) for x in params]
    else:
        return params


def load_fn():
    model = gm.nn.IntWrapper(model=gm.nn.Gemma3_1B(), dtype=jnp.int8)

    if not os.path.exists("quantized_params.msgpack"):
        params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)
        # Rename all "w" keys to "kernel"
        # params = rename_w_to_kernel(params)
        # Now quantize all "kernel" weights
        params = peft.quantize(params, method="INT8", in_place_keys=True, checkpoint_kernel_key="w")
        with open("quantized_params.msgpack", "wb") as f:
            f.write(serialization.to_bytes(params))
    with open("quantized_params.msgpack", "rb") as f:
        params = serialization.from_bytes(None, f.read())

    tokenizer = gm.text.Gemma3Tokenizer(custom_tokens={0: "<pos>", 1: "<hp>"})

    return model, params, tokenizer

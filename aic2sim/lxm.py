# %% Imports
import networkx as nx
import jax.numpy as jnp
from jax import tree, lax, debug
from jaxtyping import Array
import equinox as eqx
import parabellum as pb
import pydot
from aic2sim.utils import chess_to_int, alpha_to_int, bt_to_int, nato_to_int
from aic2sim.types import Plan, Tree
from ollama import chat


def chat_fn(messages):
    user_says = input("User: ")
    messages.append({"role": "user", "content": user_says})
    response = chat(model="deepseek-r1", messages=messages, stream=True, think=False)  # , system="you're name is hall")
    assistant_response = ""
    for chunk in response:
        content = chunk["message"]["content"]
        assistant_response += content
        print(content, end="", flush=True)
    print()
    messages.append({"role": "assistant", "content": assistant_response})
    return messages


def play_fn(scene, state, marks, messages):
    messages = [m for m in messages if m["role"] != "state"]  # remove previous state
    messages.append({"role": "state", "content": obs_fn(scene, state, marks)})  # add current state
    messages = chat_fn(messages)
    return messages


# evaluate plan
def obs_fn(cfg: pb.types.Config, state: pb.types.State, marks):  # obs function for lxm (NOT units)
    return f"raster_map: {cfg.map}\nunit_coord: {state.pos}\nunit_teams: {cfg.teams}"


# Parse plan
def str_to_plan(dot_str, scene, team):  # plan for one team
    G = nx.DiGraph(nx.nx_pydot.from_pydot(pydot.graph_from_dot_data(dot_str)[0]))  # type: ignore
    nodes = {node: tuple(data.keys()) for node, data in G.nodes(data=True)}
    return tree.map(lambda *x: jnp.stack(x), *tuple(map(lambda x: node_to_step(scene, team, G, *x), nodes.items())))  # type: ignore


def node_to_step(cfg: pb.types.Config, team, G, node, desc):
    move = jnp.array(desc[1] == "move")
    units = (jnp.tile(jnp.eye(3) == 1, cfg.length)[:, : cfg.length][nato_to_int[desc[0]]] * cfg.teams == team) * 1
    coord = jnp.array(chess_to_int[desc[2]])
    btidx = jnp.array(bt_to_int[desc[3]])
    idxs = jnp.int8([alpha_to_int[e[0]] for e in G.edges() if e[1] == node])
    parent = jnp.zeros(len(G)).at[idxs].set(1)
    return Plan(units=units, move=move, coord=coord, btidx=btidx, parent=jnp.int32(parent))

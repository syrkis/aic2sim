# %% api.py
#   This file contains the FastAPI server that serves the Parabellum environment.
# by: Noah Syrkis

# Imports
import uuid
from dataclasses import asdict, replace
from functools import partial

from ollama import chat
import cv2
import jax.numpy as jnp
import numpy as np
import parabellum as pb
from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# from fastapi.responses import StreamingResponse
import asyncio

# from fastapi import WebSocket, WebSocketDisconnect
from jax import random, tree, vmap
from omegaconf import DictConfig

import aic2sim as a2s


# Configure CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


with open("data/bts.txt", "r") as f:
    roe_str = f.read().strip()  # rules of engagement

with open("data/plan.txt", "r") as f:
    dot_str = f.read().strip().split("---")[0].strip()

with open("data/prompt.txt", "r") as f:
    llm_str = f.read().strip()

with open("data/info.txt", "r") as f:
    info_str = f.read().strip()


# %% Globals
games = {}
fps = 10
n_steps = 100
sleep_time = 1 / fps

# Config
loc = dict(place="Palazzo della Civiltà Italiana, Rome, Italy", size=64)
red = dict(infantry=6, armor=6, airplane=6)
blue = dict(infantry=6, armor=6, airplane=6)
cfg = DictConfig(dict(steps=100, knn=4, blue=blue, red=red) | loc)


env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
rng, key = random.split(random.PRNGKey(0))
bts = a2s.dsl.bts_fn(roe_str)
action_fn = vmap(a2s.act.action_fn, in_axes=(0, 0, 0, None, None, None, 0))
# targets = jnp.int32(jnp.arange(6).repeat(env.num_units // 6)).flatten()
targets = random.randint(rng, (env.num_units,), 0, 6)
plan = tree.map(lambda *x: jnp.stack(x), *tuple(map(partial(a2s.lxm.str_to_plan, dot_str, scene), (-1, 1))))  # type: ignore


def step_fn(rng, env, scene, obs: pb.types.Obs, state: pb.types.State, plan: a2s.types.Plan, gps, targets):
    rngs = random.split(rng, env.num_units)
    behavior = a2s.lxm.plan_fn(rng, bts, plan, state, scene)  # perhaps only update plan every m steps
    action = action_fn(rngs, obs, behavior, env, scene, gps, targets)
    obs, state = env.step(rng, scene, state, action)
    return (obs, state), (state, action)


# %% End points
@app.get("/init/{place}")
def init(place: str):  # should inlcude settings from frontend
    game_id = str(uuid.uuid4())
    rng = random.PRNGKey(0)
    step = partial(step_fn, rng, env, scene)
    gps = tree.map(jnp.zeros_like, a2s.gps.gps_fn(scene, jnp.int32(jnp.zeros((6, 2)))))
    game = a2s.types.Game([rng], env, scene, step, gps, [], [])  # <- state_seq list
    games[game_id] = game
    terrain = cv2.resize(np.array(scene.terrain.building), dsize=(100, 100)).tolist()
    teams = scene.unit_teams.tolist()
    marks = {k: v for k, v in zip(a2s.utils.chess_to_int, gps.marks.tolist())}
    return {"game_id": game_id, "terrain": terrain, "size": cfg.size, "teams": teams, "marks": marks}


@app.get("/reset/{game_id}")
def reset(game_id: str):
    rng, key = random.split(games[game_id].rng[-1])
    obs, state = games[game_id].env.reset(rng=key, scene=games[game_id].scene)
    games[game_id].step_seq.append(a2s.types.Step(rng, obs, state, None))
    games[game_id].rng.append(rng)
    return {"state": asdict(tree.map(lambda x: x.tolist(), state)) | {"step": len(games[game_id].step_seq)}}


@app.get("/step/{game_id}")
def step(game_id: str):
    rng, key = random.split(games[game_id].step_seq[-1].rng)
    obs, state = games[game_id].step_seq[-1].obs, games[game_id].step_seq[-1].state
    (obs, state), (state, action) = games[game_id].step_fn(obs, state, plan, games[game_id].gps, targets)
    games[game_id].step_seq.append(a2s.types.Step(rng, obs, state, action))
    return {"state": asdict(tree.map(lambda x: x.tolist(), state)) | {"step": len(games[game_id].step_seq)}}


@app.post("/close/{game_id}")
async def close(game_id: str):
    del games[game_id]


@app.post("/marks/{game_id}")
async def marks(game_id: str, marks: list = Body(...)):
    gps = a2s.gps.gps_fn(scene, jnp.int32(jnp.array(marks))[:, ::-1])
    games[game_id] = replace(games[game_id], gps=gps)


@app.websocket("/chat/ws")
async def chat_websocket(websocket: WebSocket, game_id: str | None = None):
    """WebSocket endpoint for real-time chat streaming"""
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            print(f"Received: {data}")

            # Build message history
            messages = [
                {"role": "system", "content": llm_str},
                {"role": "assistant", "content": info_str},
            ]

            # Add existing messages from game if game_id is provided and exists
            if game_id and game_id in games:
                messages.extend(games[game_id].messages)
                messages.append(a2s.lxm.obs_fn(scene, games[game_id].state[-1], marks))

            # Add current user message
            user_message = {"role": "user", "content": data}
            messages.append(user_message)

            # Stream response back to client
            stream = chat(
                model="deepseek-r1",
                messages=messages,
                stream=True,
                think=False,
            )

            # Collect the assistant's response
            assistant_response = ""
            for chunk in stream:
                content = chunk["message"]["content"]
                if content:
                    print(f"Chunk: '{content}'")
                    assistant_response += content

                    # Send chunk immediately
                    await websocket.send_text(content)

                    # Force processing of the send
                    await asyncio.gather(asyncio.sleep(0))

            # Store messages in game if game_id is provided and exists
            if game_id and game_id in games:
                games[game_id].messages.extend([user_message, {"role": "assistant", "content": assistant_response}])

            # Send completion signal
            await websocket.send_text("")
            print("Stream complete")

    except WebSocketDisconnect:
        print("Disconnected")

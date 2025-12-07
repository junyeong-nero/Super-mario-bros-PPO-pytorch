"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.env import JUMP_ONLY
from gym_super_mario_bros.actions import (
    COMPLEX_MOVEMENT,
    RIGHT_ONLY,
    SIMPLE_MOVEMENT,
)

from src.env import create_train_env, _unwrap_reset
from src.model import PPO


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Implementation of model described in the paper: "
            "Proximal Policy Optimization Algorithms for Contra Nes"
        )
    )
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    return args


ACTION_MAPPINGS = {
    "jump": JUMP_ONLY,
    "right": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
}


def test(opt):
    use_cuda = torch.cuda.is_available()
    actions = ACTION_MAPPINGS.get(opt.action_type, COMPLEX_MOVEMENT)
    video_path = f"{opt.output_path}/video_{opt.world}_{opt.stage}.mp4"
    model_path = f"{opt.saved_path}/ppo_super_mario_bros_{opt.world}_{opt.stage}"
    env = create_train_env(
        opt.world,
        opt.stage,
        actions,
        video_path,
    )
    model = PPO(env.observation_space.shape[0], len(actions))
    if use_cuda:
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    state_np, _ = _unwrap_reset(env.reset())
    state = torch.from_numpy(np.asarray(state_np))
    while True:
        if use_cuda:
            state = state.cuda()
        logits, value = model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            state_np, _ = _unwrap_reset(env.reset())
            state = torch.from_numpy(np.asarray(state_np, dtype=np.float32))
            continue
        done = terminated or truncated
        state = torch.from_numpy(np.asarray(state, dtype=np.float32))
        frame = env.render()
        # If we get raw frames (rgb_array), show them via OpenCV so gameplay
        # is visible.
        if isinstance(frame, np.ndarray):
            cv2.imshow("Mario", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Fallback for environments that handle their own rendering.
            _ = frame
        if info["flag_get"]:
            print(f"World {opt.world} stage {opt.stage} completed")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    opt = get_args()
    test(opt)

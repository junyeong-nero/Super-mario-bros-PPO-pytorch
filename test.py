"""
Super Mario Bros PPO Inference Script
Original Author: Viet Nguyen <nhviet1009@gmail.com>
Refactored for clarity and performance.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gym_super_mario_bros.actions import (
    COMPLEX_MOVEMENT,
    RIGHT_ONLY,
    SIMPLE_MOVEMENT,
)

from src.env import create_train_env, _unwrap_reset
from src.model import PPO


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parses command line arguments."""
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


JUMP_ONLY = [["right"], ["right", "A"]]

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
    else:
        model.load_state_dict(torch.load(model_path))
        model.to(device)

    model.eval()
    return model


def process_state(state: np.ndarray, device: torch.device) -> torch.Tensor:
    """Converts a numpy state to a torch tensor on the correct device."""
    state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32))
    return state_tensor.to(device)


# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------


def run_inference(args: argparse.Namespace):
    """Main loop for running the environment with the trained model."""

    # Setup paths
    saved_path = Path(args.saved_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    video_filename = output_path / f"video_{args.world}_{args.stage}.mp4"
    model_filename = saved_path / f"ppo_super_mario_bros_{args.world}_{args.stage}"

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Setup Environment
    actions = ACTION_MAPPINGS[args.action_type]
    env = create_mario_environment(
        args.world,
        args.stage,
        actions,
        args.skip_frame,
        str(video_filename),  # Env likely expects string, not Path object
    )

    # Load Model
    model = load_model(
        model_filename, env.observation_space.shape[0], len(actions), device
    )

    # Initial Reset
    # Note: _unwrap_reset is technically private, consider exposing it publicly in src.env
    state_np, _ = _unwrap_reset(env.reset())
    state = process_state(state_np, device)

    print("Starting inference... Press 'q' to quit.")

    try:
        while True:
            # Use no_grad for inference to reduce memory usage and increase speed
            with torch.no_grad():
                logits, _ = model(state)
                policy = F.softmax(logits, dim=1)
                action = torch.argmax(policy).item()

            # Step Environment
            state_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render
            frame = env.render()
            if isinstance(frame, np.ndarray):
                # Convert RGB (Gym) to BGR (OpenCV)
                cv2.imshow(
                    "Super Mario Bros PPO", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quitting...")
                    break

            # Handle Reset or Continue
            if done:
                print("Episode finished. Resetting...")
                state_np, _ = _unwrap_reset(env.reset())
                state = process_state(state_np, device)
            else:
                state = process_state(state_next, device)

            # Check for Stage Completion flags
            if info.get("flag_get"):
                print(f"World {args.world} Stage {args.stage} Completed!")
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Resources released.")


if __name__ == "__main__":
    opt = get_args()
    run_inference(opt)

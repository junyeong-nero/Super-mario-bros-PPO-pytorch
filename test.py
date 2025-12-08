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
from gym.spaces import Space

# Limit OMP threads before importing libraries that might use them
os.environ["OMP_NUM_THREADS"] = "1"

# Local imports
from src.env import create_mario_environment
from src.env import ACTION_MAPPINGS
from src.model import PPO
from src.schema import SuperMarioObs
from src.utils import *

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="PPO Inference for Super Mario Bros (NES)"
    )
    parser.add_argument("--world", type=int, default=1, help="World number (1-8)")
    parser.add_argument("--stage", type=int, default=1, help="Stage number (1-4)")
    parser.add_argument(
        "--action_type",
        type=str,
        default="simple",
        choices=ACTION_MAPPINGS.keys(),
        help="Action space complexity",
    )
    parser.add_argument(
        "--saved_path",
        type=str,
        default="trained_models",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output",
        help="Directory to save video output",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def load_model(
    model_path: Path, input_dim: int, output_dim: int, device: torch.device
) -> PPO:
    """Initializes the PPO model and loads weights."""
    model = PPO(input_dim, output_dim)

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    print(f"Loading model from: {model_path}")
    if device.type == "cpu":
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
    log_filename = output_path / f"obs_{args.world}_{args.stage}.txt"

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Setup Environment
    actions = ACTION_MAPPINGS[args.action_type]
    env, monitor = create_mario_environment(
        args.world,
        args.stage,
        actions,
        str(video_filename),  # Env likely expects string, not Path object
    )

    # Load Model
    model = load_model(
        model_filename, env.observation_space.shape[0], len(actions), device
    )

    # Initial Reset
    state_np, _, _, _, _ = env.reset()
    state = preprocess_image(state_np)

    print("Starting inference... Press 'q' to quit.")

    log_file = log_filename.open("a", encoding="utf-8")

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
            state_next_np = preprocess_image(state_next)

            obs = SuperMarioObs(
                state={"image": state_next},
                image=state_next,
                info=info,
                reward={"distance": info["x_pos"], "done": done},
            )

            print(obs.to_text())

            # Handle Reset or Continue
            if done:
                print("Episode finished. Resetting...")
                state_np, _, _, _, _ = env.reset()
                state = preprocess_image(state_np)
            else:
                state = state_next_np

            # Check for Stage Completion flag
            if info.get("flag_get"):
                print(f"World {args.world} Stage {args.stage} Completed!")
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        monitor.close()
        log_file.close()
        cv2.destroyAllWindows()
        print("Resources released.")


if __name__ == "__main__":
    opt = get_args()
    run_inference(opt)

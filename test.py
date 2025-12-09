"""
Super Mario Bros PPO Inference Script
Original Author: Viet Nguyen <nhviet1009@gmail.com>
Refactored for clarity and performance.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable logging observations to JSON.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save logging files.",
    )
    parser.add_argument(
        "--log_index",
        type=int,
        default=None,
        help="Optional log index; defaults to next available number.",
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


def to_json_safe(value: Any) -> Any:
    """Converts numpy types and containers into JSON serializable objects."""
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def resolve_log_index(log_dir: Path, provided_index: int | None) -> int:
    """Returns a log index, using the next available number when not provided."""
    if provided_index is not None:
        return provided_index

    existing_indices = [
        int(path.stem) for path in log_dir.glob("*.json") if path.stem.isdigit()
    ]
    return (max(existing_indices) + 1) if existing_indices else 0


def save_logging_history(
    history: List[SuperMarioObs],
    args: argparse.Namespace,
    base_dir: Path,
) -> Path:
    """Saves observation history to logs/{world}_{stage}/{index}.json."""
    run_dir = base_dir / f"{args.world}_{args.stage}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_index = resolve_log_index(run_dir, args.log_index)
    log_path = run_dir / f"{log_index}.json"

    payload = {
        "world": args.world,
        "stage": args.stage,
        "action_type": args.action_type,
        "history": [
            {
                "time": obs.time,
                "info": to_json_safe(obs.info),
                "reward": to_json_safe(obs.reward),
                "observation": obs.to_text(),
                "objects": obs.to_dict(),
            }
            for obs in history
        ],
    }

    with log_path.open("w", encoding="utf-8") as log_file:
        json.dump(payload, log_file, indent=2)

    return log_path


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
    done = False
    state, reward, term, trunc, info = env.reset()

    print("Starting inference... Press 'q' to quit.")

    logging_enabled = args.log
    history = []

    try:
        while True:
            # Use no_grad for inference to reduce memory usage and increase speed
            with torch.no_grad():
                logits, _ = model(preprocess_image(state))
                policy = F.softmax(logits, dim=1)
                action = torch.argmax(policy).item()

            # logging current state & action
            if logging_enabled:
                obs = SuperMarioObs(
                    state={"image": state},
                    image=state,
                    info=info,
                    reward={"action": action, "distance": info["x_pos"], "done": done},
                )
                obs.time = datetime.now().strftime("%Y%m%d_%H%M%S")
                history.append(obs)

            # Step Environment
            state_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Handle Reset or Continue
            if done:
                # Check for Stage Completion flag
                if info.get("flag_get"):
                    print(f"World {args.world} Stage {args.stage} Completed!")
                    break

                print("Episode finished. Resetting...")
                state, reward, term, trunc, info = env.reset()
            else:
                state = state_next

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        monitor.close()
        if logging_enabled and history:
            log_path = save_logging_history(history, args, Path(args.log_dir))
            print(f"Saved log to: {log_path}")
        cv2.destroyAllWindows()
        print("Resources released.")


if __name__ == "__main__":
    opt = get_args()
    run_inference(opt)

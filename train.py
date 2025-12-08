"""
@author: Viet Nguyen <nhviet1009@gmail.com>
Refactored for readability and maintainability.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# Set OMP threads before importing other heavy libraries if necessary
os.environ["OMP_NUM_THREADS"] = "1"

from src.env import ACTION_MAPPINGS, _unwrap_reset, create_mario_environment
from src.model import PPO


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO Implementation for Super Mario Bros"
    )

    # Environment settings
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument("--skip_frame", type=int, default=4)

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--tau", type=float, default=1.0, help="GAE parameter")
    parser.add_argument("--beta", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clip parameter")

    # Training settings
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5_000_000)
    parser.add_argument(
        "--num_processes", type=int, default=1, help="Set to 1 for this implementation"
    )
    parser.add_argument(
        "--save_interval", type=int, default=50, help="Steps between saves"
    )

    # Paths
    parser.add_argument(
        "--log_path", type=str, default="tensorboard/ppo_super_mario_bros"
    )
    parser.add_argument("--saved_path", type=str, default="trained_models")

    return parser.parse_args()


def to_tensor(state_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Helper to convert numpy state to tensor."""
    return torch.from_numpy(np.asarray(state_np, dtype=np.float32)).to(device)


def compute_gae(
    next_value: torch.Tensor,
    rewards: List[torch.Tensor],
    dones: List[torch.Tensor],
    values: List[torch.Tensor],
    gamma: float,
    tau: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Computes Generalized Advantage Estimation (GAE).
    """
    gae = torch.zeros(1, device=device)
    returns = []

    # Iterate backwards through the trajectory
    for step_value, step_reward, step_done in reversed(
        list(zip(values, rewards, dones))
    ):
        delta = (
            step_reward
            + gamma * next_value.detach() * (1 - step_done)
            - step_value.detach()
        )
        gae = delta + gamma * tau * gae * (1 - step_done)

        # Calculate return: GAE + Value = (Return - Value) + Value = Return
        returns.insert(0, gae + step_value.detach())
        next_value = step_value

    return torch.cat(returns)


def generate_batches(
    batch_size: int, minibatch_size: int, device: torch.device
) -> Iterator[torch.Tensor]:
    """Generates random indices for mini-batch updates."""
    indices = torch.randperm(batch_size, device=device)
    for i in range(0, batch_size, minibatch_size):
        yield indices[i : i + minibatch_size]


def update_policy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    args: argparse.Namespace,
) -> float:
    """
    Performs the PPO update loop.
    Returns the total loss for logging.
    """
    total_loss_sum = 0.0
    batch_size = states.size(0)
    num_updates = 0

    for _ in range(args.num_epochs):
        for batch_indices in generate_batches(
            batch_size, args.batch_size, states.device
        ):
            # Extract mini-batch
            b_states = states[batch_indices]
            b_actions = actions[batch_indices]
            b_old_log_probs = old_log_probs[batch_indices]
            b_returns = returns[batch_indices]
            b_advantages = advantages[batch_indices]

            # Forward pass
            logits, values = model(b_states)
            policy_dist = Categorical(logits=logits)
            new_log_probs = policy_dist.log_prob(b_actions)
            entropy = policy_dist.entropy().mean()

            # PPO Ratio
            ratio = torch.exp(new_log_probs - b_old_log_probs)

            # Actor Loss (Clipped)
            surr1 = ratio * b_advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - args.epsilon, 1.0 + args.epsilon)
                * b_advantages
            )
            actor_loss = -torch.mean(torch.min(surr1, surr2))

            # Critic Loss
            critic_loss = F.smooth_l1_loss(values.squeeze(), b_returns)

            # Total Loss
            loss = actor_loss + critic_loss - args.beta * entropy

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss_sum += loss.item()
            num_updates += 1

    return total_loss_sum / num_updates if num_updates > 0 else 0.0


def train(args: argparse.Namespace):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seeding
    torch.manual_seed(123)
    if device.type == "cuda":
        torch.cuda.manual_seed(123)

    # File Setup
    log_path = Path(args.log_path)
    if log_path.exists():
        shutil.rmtree(log_path)
    log_path.mkdir(parents=True, exist_ok=True)

    save_path = Path(args.saved_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Environment
    if args.num_processes != 1:
        print("Warning: Multiprocessing override enabled. Setting num_processes to 1.")

    actions_space = ACTION_MAPPINGS.get(args.action_type, ACTION_MAPPINGS["complex"])
    env = create_mario_environment(
        args.world, args.stage, actions_space, args.skip_frame
    )

    num_states = env.observation_space.shape[0]
    num_actions = len(actions_space)

    # Model & Optimizer
    model = PPO(num_states, num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_base = save_path / f"ppo_super_mario_bros_{args.world}_{args.stage}"

    # --- Training Loop ---
    state_np, _ = _unwrap_reset(env.reset())
    state = to_tensor(state_np, device)

    total_steps = 0
    curr_episode = 0
    last_save_step = 0

    print("Starting training...")

    while total_steps < args.num_global_steps:
        curr_episode += 1

        # Buffers
        log_probs_buffer = []
        actions_buffer = []
        values_buffer = []
        states_buffer = []
        rewards_buffer = []
        dones_buffer = []

        # 1. Collect Trajectory
        for _ in range(args.num_local_steps):
            states_buffer.append(state)

            with torch.no_grad():
                logits, value = model(state)
                policy_dist = Categorical(logits=logits)
                action = policy_dist.sample()
                log_prob = policy_dist.log_prob(action)

            values_buffer.append(value.view(1))
            actions_buffer.append(action)
            log_probs_buffer.append(log_prob)

            # Environment Step
            state_np, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            state = to_tensor(state_np, device)

            rewards_buffer.append(
                torch.tensor([reward], dtype=torch.float32, device=device)
            )
            dones_buffer.append(
                torch.tensor([float(done)], dtype=torch.float32, device=device)
            )

            total_steps += 1

            if done:
                state_np, _ = _unwrap_reset(env.reset())
                state = to_tensor(state_np, device)

            if total_steps >= args.num_global_steps:
                break

        # 2. Compute Returns & Advantages
        with torch.no_grad():
            _, next_value = model(state)
            next_value = next_value.view(1)

        # Detach and concatenate buffers
        returns = compute_gae(
            next_value,
            rewards_buffer,
            dones_buffer,
            values_buffer,
            args.gamma,
            args.tau,
            device,
        )

        # Convert lists to tensors
        states_tensor = torch.cat(states_buffer)
        actions_tensor = torch.cat(actions_buffer)
        old_log_probs_tensor = torch.cat(log_probs_buffer).detach()
        values_tensor = torch.cat(values_buffer).detach()

        # Advantage Normalization (Optional but recommended, though not in original code)
        advantages = returns - values_tensor

        # 3. Update Policy
        avg_loss = update_policy(
            model,
            optimizer,
            states_tensor,
            actions_tensor,
            old_log_probs_tensor,
            returns,
            advantages,
            args,
        )

        print(
            f"Episode: {curr_episode} | Steps: {total_steps}/{args.num_global_steps} | Avg Loss: {avg_loss:.4f}"
        )

        # 4. Save Model
        if (
            total_steps - last_save_step >= args.save_interval
        ) or total_steps >= args.num_global_steps:
            torch.save(model.state_dict(), str(checkpoint_base))
            last_save_step = total_steps
            print(f"Model saved to {checkpoint_base}")


if __name__ == "__main__":
    opt = get_args()
    train(opt)

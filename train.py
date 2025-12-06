"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import argparse
import os
import shutil

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.env import ACTION_MAPPINGS, _unwrap_reset, create_train_env
from src.model import PPO


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Implementation of model described in the paper: "
            "Proximal Policy Optimization Algorithms for Super Mario Bros"
        )
    )
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="discount factor for rewards"
    )
    parser.add_argument("--tau", type=float, default=1.0, help="parameter for GAE")
    parser.add_argument("--beta", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="parameter for Clipped Surrogate Objective",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument(
        "--save_interval", type=int, default=50, help="Number of steps between savings"
    )
    parser.add_argument(
        "--max_actions",
        type=int,
        default=200,
        help="Maximum repetition steps in test phase",
    )
    parser.add_argument(
        "--log_path", type=str, default="tensorboard/ppo_super_mario_bros"
    )
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def train(opt):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(123)
    if use_cuda:
        torch.cuda.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    if opt.num_processes != 1:
        print("Multiprocessing disabled; overriding num_processes to 1.")
    actions_space = ACTION_MAPPINGS.get(opt.action_type, ACTION_MAPPINGS["complex"])
    env = create_train_env(opt.world, opt.stage, actions_space)
    num_states = env.observation_space.shape[0]
    num_actions = len(actions_space)
    model = PPO(num_states, num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    base_checkpoint = f"{opt.saved_path}/ppo_super_mario_bros_{opt.world}_{opt.stage}"

    state_np, _ = _unwrap_reset(env.reset())
    state = torch.from_numpy(np.asarray(state_np, dtype=np.float32)).to(device)
    max_steps = int(opt.num_global_steps)
    curr_episode = 0
    total_steps = 0
    last_save_step = 0
    while total_steps < max_steps:
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        for _ in range(opt.num_local_steps):
            states.append(state)
            logits, value = model(state)
            values.append(value.view(1))
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policies.append(old_m.log_prob(action))

            state_np, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            state = torch.from_numpy(np.asarray(state_np, dtype=np.float32)).to(device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
            done_tensor = torch.tensor(
                [float(done)], dtype=torch.float32, device=device
            )
            rewards.append(reward_tensor)
            dones.append(done_tensor)
            total_steps += 1
            if done:
                state_np, _ = _unwrap_reset(env.reset())
                state = torch.from_numpy(np.asarray(state_np, dtype=np.float32)).to(
                    device
                )
            if total_steps >= max_steps:
                break

        _, next_value = model(state)
        next_value = next_value.view(1)
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        gae = torch.zeros(1, device=device)
        returns = []
        for value, reward, done in reversed(list(zip(values, rewards, dones))):
            gae = gae * opt.gamma * opt.tau
            gae = (
                gae
                + reward
                + opt.gamma * next_value.detach() * (1 - done)
                - value.detach()
            )
            next_value = value
            returns.append(gae + value)
        returns = torch.cat(list(reversed(returns))).detach()
        advantages = returns - values
        batch_size = states.size(0)
        for _ in range(opt.num_epochs):
            indices = torch.randperm(batch_size, device=device)
            for j in range(opt.batch_size):
                start = int(j * batch_size / opt.batch_size)
                end = int((j + 1) * batch_size / opt.batch_size)
                if start >= end:
                    continue
                batch_indices = indices[start:end]
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(
                    torch.min(
                        ratio * advantages[batch_indices],
                        torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon)
                        * advantages[batch_indices],
                    )
                )
                critic_loss = F.smooth_l1_loss(returns[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        print(
            f"Episode: {curr_episode}. Total loss: {total_loss}. Steps: {total_steps}"
        )
        if (
            total_steps - last_save_step >= opt.save_interval
            or total_steps >= max_steps
        ):
            torch.save(model.state_dict(), base_checkpoint)
            # torch.save(model.state_dict(), f"{base_checkpoint}_{total_steps}")
            last_save_step = total_steps


if __name__ == "__main__":
    opt = get_args()
    train(opt)

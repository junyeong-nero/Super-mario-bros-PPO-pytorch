"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from src.env import create_mario_environment, _unwrap_reset
from src.model import PPO


def eval(opt, global_model, num_states, num_actions):
    torch.manual_seed(123)
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = create_mario_environment(opt.world, opt.stage, actions)
    local_model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.eval()
    state_np, _ = _unwrap_reset(env.reset())
    state_np = np.asarray(state_np)
    state = torch.from_numpy(state_np)
    if torch.cuda.is_available():
        state = state.cuda()
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        logits, value = local_model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = np.asarray(state, dtype=np.float32)

        # Uncomment following lines if you want to save model whenever level is completed
        # if info["flag_get"]:
        #     print("Finished")
        #     torch.save(local_model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_step))

        env.render()
        actions.append(action)
        if (
            curr_step > opt.num_global_steps
            or actions.count(actions[0]) == actions.maxlen
        ):
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state_np, _ = _unwrap_reset(env.reset())
            state = np.asarray(state_np, dtype=np.float32)
        state = torch.from_numpy(state)
        if torch.cuda.is_available():
            state = state.cuda()

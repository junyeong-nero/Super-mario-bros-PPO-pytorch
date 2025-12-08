"""
@author: Viet Nguyen <nhviet1009@gmail.com>
@refactor: Beautified and Modernized
"""

import subprocess as sp
from typing import Any, List, Tuple, Dict, Optional, Union
import multiprocessing as mp

import cv2
import gym
import numpy as np
import gym_super_mario_bros
from gym import Wrapper
from gym.spaces import Box
from gym.wrappers import FrameStack, TransformObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from src.config import TARGET_SHAPE
from src.utils import *

# --- NumPy 2.0 Compatibility Patch ---
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# --- Constants ---

JUMP_ONLY = [
    ["right"],
    ["right", "A"],
    ["right", "A", "A"],
    ["right", "A", "A", "A"],
    ["right", "A", "A", "A", "A"],
    ["right", "A", "A", "A", "A", "A"],
    ["right", "A", "A", "A", "A", "A", "A"],
]

ACTION_MAPPINGS = {
    "jump": JUMP_ONLY,
    "right": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
}


# --- Utility Functions ---


def normalize_observation(observation: np.ndarray) -> np.ndarray:
    """Scale observation to [0, 1] for TransformObservation."""
    return (observation / 255.0).astype(np.float32)


# --- Wrappers and Classes ---

from src.monitor import Monitor


class SkipFrame(Wrapper):
    """Repeat the same action for `skip` frames and accumulate rewards."""

    def __init__(self, env: gym.Env, skip: int, monitor=None):
        super().__init__(env)
        self._skip = skip
        self.monitor = monitor

    def step(self, action: int):
        total_reward = 0.0
        last_obs, info = None, {}
        terminated = False
        truncated = False

        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            if self.monitor:
                self.monitor.record(self.env)

            total_reward += reward
            last_obs = obs
            terminated = term or terminated
            truncated = trunc or truncated

            if terminated or truncated:
                break

        return last_obs, total_reward, terminated, truncated, info


class CustomReward(Wrapper):
    """
    Custom wrapper to handle specialized reward logic, rendering,
    and specific penalties for World 7-4 and 4-4.
    """

    def __init__(
        self,
        env: gym.Env,
        world: int,
        stage: int,
        actions,
        monitor: Optional[Monitor] = None,
    ):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, *TARGET_SHAPE))
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage
        self.is_jump_only = actions == JUMP_ONLY
        self.monitor = monitor

    def _apply_level_constraints(self, x_pos: int, y_pos: int) -> Tuple[float, bool]:
        """Checks specific coordinates for worlds 7-4 and 4-4 to apply penalties."""
        penalty = 0.0
        force_done = False

        # Logic for World 7-4
        if self.world == 7 and self.stage == 4:
            if (
                (506 <= x_pos <= 832 and y_pos > 127)
                or (832 < x_pos <= 1064 and y_pos < 80)
                or (1113 < x_pos <= 1464 and y_pos < 191)
                or (1579 < x_pos <= 1943 and y_pos < 191)
                or (1946 < x_pos <= 1964 and y_pos >= 191)
                or (1984 < x_pos <= 2060 and (y_pos >= 191 or y_pos < 127))
                or (2114 < x_pos < 2440 and y_pos < 191)
                or x_pos < self.current_x - 500
            ):
                penalty = -50.0
                force_done = True

        # Logic for World 4-4
        if self.world == 4 and self.stage == 4:
            if (x_pos <= 1500 and y_pos < 127) or (
                1588 <= x_pos < 2380 and y_pos >= 127
            ):
                penalty = -50.0
                force_done = True

        return penalty, force_done

    def step(self, action: int):
        state, reward, terminated, truncated, info = None, 0.0, False, False, {}

        # Handle custom action logic (0 is simple step, others repeat action 1)
        if self.is_jump_only:

            # - Jump Levels *(values based on flat ground jumps)*:
            # - Level 0: +0 in x, +0 in y (No jump, just walk)
            # - Level 1: +42 in x, +35 in y
            # - Level 2: +56 in x, +46 in y
            # - Level 3: +63 in x, +53 in y
            # - Level 4: +70 in x, +60 in y
            # - Level 5: +77 in x, +65 in y
            # - Level 6: +84 in x, +68 in y

            if action == 0:
                state, reward, terminated, truncated, info = self.env.step(0)
                if self.monitor:
                    self.monitor.record(self.env)

            else:
                # Execute jump/action sequence
                for _ in range(action):
                    state, reward, terminated, truncated, info = self.env.step(1)
                    if self.monitor:
                        self.monitor.record(self.env)
                    if terminated or truncated:
                        break

                # Handle "On Air" logic
                if not (terminated or truncated):
                    on_air = True
                    mario_y_history = []
                    while on_air:
                        state, reward, term_step, trunc_step, info = self.env.step(0)
                        if self.monitor:
                            self.monitor.record(self.env)

                        terminated = terminated or term_step
                        truncated = truncated or trunc_step

                        if terminated or truncated:
                            break

                        mario_y_history.append(info["y_pos"])
                        # Check if Mario has been at the same Y height for 3 frames (landed)
                        if len(mario_y_history) >= 3:
                            if (
                                mario_y_history[-3]
                                == mario_y_history[-2]
                                == mario_y_history[-1]
                            ):
                                on_air = False

        else:
            state, reward, terminated, truncated, info = self.env.step(action)
            if self.monitor:
                self.monitor.record(self.env)

        # Processing State and Reward
        reward += (info["score"] - self.curr_score) / 40.0
        self.curr_score = info["score"]

        done = terminated or truncated
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50

        # Apply specific level penalties
        penalty, force_done = self._apply_level_constraints(
            info["x_pos"], info["y_pos"]
        )
        reward += penalty
        if force_done:
            done = True
            terminated = True

        self.current_x = info["x_pos"]
        return state, reward / 10.0, terminated, truncated, info

    def reset(self):

        # reset environment
        self.env.reset()
        self.curr_score = 0
        self.current_x = 40

        start_frame = 1
        for _ in range(start_frame):
            obs, reward, term, trunc, info = self.env.step(0)

        return obs, reward, term, trunc, info


# --- Environment Factories ---


def create_mario_environment(
    world: int,
    stage: int,
    actions: List[List[str]],
    output_path: Optional[str] = None,
    render_mode: str = "human",  # rgb_array
) -> gym.Env:
    """Creates and wraps the training environment."""
    env = gym_super_mario_bros.make(
        f"SuperMarioBros-{world}-{stage}-v1",
        apply_api_compatibility=True,
        render_mode=render_mode,
    )

    monitor = Monitor(256, 240, output_path) if output_path else None

    env = JoypadSpace(env, actions)
    env = SkipFrame(env, skip=4, monitor=monitor)
    env = TransformObservation(env, f=normalize_observation)
    env = FrameStack(env, num_stack=1)
    env = CustomReward(env, world, stage, actions, monitor)

    return env

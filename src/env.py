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

# --- NumPy 2.0 Compatibility Patch ---
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# --- Constants ---
TARGET_SHAPE = (84, 84)

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


def process_frame(frame: Optional[np.ndarray]) -> np.ndarray:
    """Converts a frame to grayscale, resizes it, and normalizes pixel values."""
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, TARGET_SHAPE)[None, :, :] / 255.0
        return frame.astype(np.float32)
    return np.zeros((1, *TARGET_SHAPE), dtype=np.float32)


def normalize_observation(observation: np.ndarray) -> np.ndarray:
    """Scale observation to [0, 1] for TransformObservation."""
    return (observation / 255.0).astype(np.float32)


def _to_numpy(obs: Any) -> np.ndarray:
    """Ensure observation is a NumPy array (handles LazyFrames)."""
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32, copy=False)
    return np.asarray(obs, dtype=np.float32)


def _unwrap_reset(result: Any) -> Tuple[Any, Dict]:
    """Normalize reset outputs across Gym / Gymnasium API versions."""
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _unwrap_step(result: Any) -> Tuple[Any, float, bool, bool, Dict]:
    """Normalize step outputs to (state, reward, terminated, truncated, info)."""
    if len(result) == 5:
        return result
    # Legacy 4-tuple: (state, reward, done, info)
    state, reward, done, info = result
    return state, reward, done, False, info


# --- Wrappers and Classes ---


class Monitor:
    """Records the environment using ffmpeg."""

    def __init__(self, width: int, height: int, saved_path: str):
        self.command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}X{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            "60",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "mpeg4",
            saved_path,
        ]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            print("Error: ffmpeg not found. Video recording disabled.")
            self.pipe = None

    def record(self, image_array: np.ndarray) -> None:
        if self.pipe is not None:
            self.pipe.stdin.write(image_array.tobytes())


class SkipFrame(Wrapper):
    """Repeat the same action for `skip` frames and accumulate rewards."""

    def __init__(self, env: gym.Env, skip: int):
        super().__init__(env)
        self._skip = skip

    def step(self, action: int):
        total_reward = 0.0
        last_obs, info = None, {}
        terminated = False
        truncated = False

        for _ in range(self._skip):
            obs, reward, term, trunc, info = _unwrap_step(self.env.step(action))
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
                state, reward, terminated, truncated, info = _unwrap_step(
                    self.env.step(0)
                )
            else:
                # Execute jump/action sequence
                for _ in range(action):
                    state, reward, terminated, truncated, info = _unwrap_step(
                        self.env.step(1)
                    )
                    if terminated or truncated:
                        break

                # Handle "On Air" logic
                if not (terminated or truncated):
                    on_air = True
                    mario_y_history = []
                    while on_air:
                        state, reward, term_step, trunc_step, info = _unwrap_step(
                            self.env.step(0)
                        )
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
            state, reward, terminated, truncated, info = _unwrap_step(
                self.env.step(action)
            )

        # Rendering and Recording
        self.env.render()
        if self.monitor:
            self.monitor.record(state)

        # Processing State and Reward
        processed_state = process_frame(state)
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

        return processed_state, reward / 10.0, terminated, truncated, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        obs, info = _unwrap_reset(self.env.reset())
        return process_frame(obs), info


# --- Environment Factories ---


def create_mario_environment(
    world: int,
    stage: int,
    actions: List[List[str]],
    output_path: Optional[str] = None,
    render_mode: str = "human",
) -> gym.Env:
    """Creates and wraps the training environment."""
    env = gym_super_mario_bros.make(
        f"SuperMarioBros-{world}-{stage}-v1",
        apply_api_compatibility=True,
        render_mode=render_mode,
    )

    monitor = Monitor(256, 240, output_path) if output_path else None

    is_jump_only = actions == JUMP_ONLY

    env = JoypadSpace(env, actions)
    env = SkipFrame(env, skip=4)
    env = CustomReward(env, world, stage, actions, monitor)
    env = TransformObservation(env, f=normalize_observation)
    env = FrameStack(env, num_stack=1)

    return env

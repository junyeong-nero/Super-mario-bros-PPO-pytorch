"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import cv2
import numpy as np
import subprocess as sp

# NumPy 2.0 removed the bool8 alias that old Gym still references.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gym
from gym import Wrapper as GymWrapper
from gym.spaces import Box
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

JUMP_ONLY = [
    ["right"],
    ["right", "A"],
    ["right", "A", "A"],
    ["right", "A", "A", "A"],
    ["right", "A", "A", "A", "A"],
    ["right", "A", "A", "A", "A", "A"],
    ["right", "A", "A", "A", "A", "A", "A"],
]

from nes_py.wrappers import JoypadSpace
import torch.multiprocessing as mp


class SkipFrame(GymWrapper):
    """Repeat the same action for `skip` frames and accumulate rewards."""

    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
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


class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            "{}X{}".format(width, height),
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
            self.pipe = None

    def record(self, image_array):
        if self.pipe is not None:
            self.pipe.stdin.write(image_array.tobytes())


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.0
        return frame.astype(np.float32)
    else:
        return np.zeros((1, 84, 84), dtype=np.float32)


def _unwrap_reset(result):
    """Normalize reset outputs across Gym / Gymnasium API versions."""
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _unwrap_step(result):
    """Normalize step outputs to (state, reward, terminated, truncated, info)."""
    if len(result) == 5:
        return result
    # Legacy 4-tuple: (state, reward, done, info)
    state, reward, done, info = result
    return state, reward, done, False, info


def normalize_observation(observation):
    """Scale observation to [0, 1] for TransformObservation (pickle-safe)."""
    return (observation / 255.0).astype(np.float32)


def _to_numpy(obs):
    """Ensure observation is a NumPy array (handles LazyFrames)."""
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32, copy=False)
    return np.asarray(obs, dtype=np.float32)


class CustomReward(GymWrapper):
    def __init__(self, env=None, world=None, stage=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        # TODO implementation n_jumps
        # state, reward, terminated, truncated, info = _unwrap_step(self.env.step(action))
        # print(action)

        state, reward, terminated, truncated, info = None, None, None, None, None
        if action == 0:
            state, reward, terminated, truncated, info = _unwrap_step(
                self.env.step(action=0)
            )
        else:
            for _ in range(action):
                state, reward, terminated, truncated, info = _unwrap_step(
                    self.env.step(action=1)
                )
                if terminated:
                    break

            on_air = True
            mario_y_history = []
            while on_air and not terminated:
                state, reward, terminated, truncated, info = self.env.step(action=0)
                if terminated:
                    break

                mario_y_history.append(info["y_pos"])
                if len(mario_y_history) >= 3:
                    if (
                        mario_y_history[-3] == mario_y_history[-2]
                        and mario_y_history[-2] == mario_y_history[-1]
                    ):
                        on_air = False

        self.env.render()
        done = terminated or truncated
        if self.monitor:
            self.monitor.record(state)
        state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.0
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        if self.world == 7 and self.stage == 4:
            if (
                (506 <= info["x_pos"] <= 832 and info["y_pos"] > 127)
                or (832 < info["x_pos"] <= 1064 and info["y_pos"] < 80)
                or (1113 < info["x_pos"] <= 1464 and info["y_pos"] < 191)
                or (1579 < info["x_pos"] <= 1943 and info["y_pos"] < 191)
                or (1946 < info["x_pos"] <= 1964 and info["y_pos"] >= 191)
                or (
                    1984 < info["x_pos"] <= 2060
                    and (info["y_pos"] >= 191 or info["y_pos"] < 127)
                )
                or (2114 < info["x_pos"] < 2440 and info["y_pos"] < 191)
                or info["x_pos"] < self.current_x - 500
            ):
                reward -= 50
                done = True
        if self.world == 4 and self.stage == 4:
            if (info["x_pos"] <= 1500 and info["y_pos"] < 127) or (
                1588 <= info["x_pos"] < 2380 and info["y_pos"] >= 127
            ):
                reward = -50
                done = True

        self.current_x = info["x_pos"]
        terminated = done or terminated
        return state, reward / 10.0, terminated, truncated, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        obs, info = _unwrap_reset(self.env.reset())
        return process_frame(obs), info


class CustomSkipFrame(GymWrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, terminated, truncated, info = _unwrap_step(
                self.env.step(action)
            )
            done = terminated or truncated
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                terminated = done
                return (
                    self.states[None, :, :, :].astype(np.float32),
                    total_reward,
                    terminated,
                    truncated,
                    info,
                )
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return (
            self.states[None, :, :, :].astype(np.float32),
            total_reward,
            False,
            False,
            info,
        )

    def reset(self):
        state, info = _unwrap_reset(self.env.reset())
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32), info


def create_train_env(world, stage, actions, output_path=None, render_mode="human"):
    env = gym_super_mario_bros.make(
        "SuperMarioBros-{}-{}-v1".format(world, stage),
        apply_api_compatibility=True,
        render_mode=render_mode,
    )
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None

    env = JoypadSpace(env, actions)
    env = SkipFrame(env, skip=4)
    env = CustomReward(env, world, stage, monitor)
    env = TransformObservation(env, f=normalize_observation)
    env = FrameStack(env, num_stack=1)

    # origianl environments
    # env = JoypadSpace(env, actions)
    # env = GymToGymnasiumAdapter(env)
    # env = CustomReward(env, world, stage, monitor)
    # env = CustomSkipFrame(env)
    return env


ACTION_MAPPINGS = {
    "jump": JUMP_ONLY,
    "right": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
}


class MultipleEnvironments:
    def __init__(self, world, stage, action_type, num_envs, output_path=None):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        actions = ACTION_MAPPINGS.get(action_type, COMPLEX_MOVEMENT)
        self.envs = [
            create_train_env(world, stage, actions, output_path=output_path)
            for _ in range(num_envs)
        ]
        self.num_states = self.envs[0].observation_space.shape[0]
        self.num_actions = len(actions)
        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index,))
            process.start()
            self.env_conns[index].close()

    def run(self, index):
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                state, reward, terminated, truncated, info = self.envs[index].step(
                    action.item()
                )
                state = _to_numpy(state)
                done = terminated or truncated
                self.env_conns[index].send((state, reward, done, info))
            elif request == "reset":
                obs, _ = _unwrap_reset(self.envs[index].reset())
                obs = _to_numpy(obs)
                self.env_conns[index].send(obs)
            else:
                raise NotImplementedError

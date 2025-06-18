import gym
from collections import deque
from gym import spaces
import torch
from preprocessing import preprocess
import numpy as np

class ImageOnlyWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_frames=8):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        self.env = env

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_frames, 84, 84), dtype=np.float32
        )

    def reset(self):
        _ = self.env.reset()
        frame = self.env.render(mode='rgb_array')
        processed = preprocess(frame)
        for _ in range(self.n_frames):
            self.frames.append(processed.clone())
        return torch.cat(list(self.frames), dim=0).numpy()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        frame = self.env.render(mode='rgb_array')
        processed = preprocess(frame)
        self.frames.append(processed)
        return torch.cat(list(self.frames), dim=0).numpy(), reward, done, info
import gym
import numpy as np
import torch
from collections import deque
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from images.image_preprocessing import preprocess

# Environment wrapper for images and states input
class CombinedWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_frames=8):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        self.env = env

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0.0, high=1.0, shape=(n_frames, 84, 84), dtype=np.float32),
            "state": env.observation_space
        })

    def reset(self):
        state = self.env.reset()
        frame = self.env.render(mode='rgb_array')
        processed = preprocess(frame)
        for _ in range(self.n_frames):
            self.frames.append(processed.clone())
        
        return {
            "image": torch.cat(list(self.frames), dim=0).numpy(),
            "state": state
        }

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        frame = self.env.render(mode='rgb_array')
        processed = preprocess(frame)
        self.frames.append(processed)

        # Save image
        """processed_img = transforms.ToPILImage()(processed)
        os.makedirs("frames", exist_ok=True)
        frame_path = os.path.join("frames", f"frame_gray.png")
        processed_img.save(frame_path)"""
        
        return {
            "image": torch.cat(list(self.frames), dim=0).numpy(),
            "state": state
        }, reward, done, info

# Feature extractor for images and states input
class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(8, 32, kernel_size=8, stride=4), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space["image"].sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        state_dim = observation_space["state"].shape[0]

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten + state_dim, 523),
            torch.nn.ReLU()
        )

    def forward(self, obs):
        image_feat = self.cnn(obs["image"])
        state_feat = obs["state"]
        return self.linear(torch.cat([image_feat, state_feat], dim=1))

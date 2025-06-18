import gym
import numpy as np
import torch
from collections import deque
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from images.image_preprocessing import preprocess

# Environment wrapper for image only input
class ImageOnlyWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_frames=8):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        self.env = env
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_frames, 84, 84), dtype=np.float32)

    def reset(self):
        self.env.reset()
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

        # Save image
        """processed_img = transforms.ToPILImage()(processed)
        os.makedirs("frames", exist_ok=True)
        frame_path = os.path.join("frames", f"frame_gray.png")
        processed_img.save(frame_path)"""

        return torch.cat(list(self.frames), dim=0).numpy(), reward, done, info

# Feature extractor for image only input
class ImageOnlyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim=512):
        super().__init__(observation_space, features_dim)

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(8, 32, kernel_size=8, stride=4), #se cambio n_frames da passare cambia il primo valore qui 
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, 512),
            torch.nn.ReLU()
        )

    def forward(self, obs):
        image_feat = self.cnn(obs)
        return self.linear(image_feat)
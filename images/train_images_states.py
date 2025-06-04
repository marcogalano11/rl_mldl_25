# Combined visual + state vector PPO training and evaluation

from PIL import Image
import gym
from mujoco_py import GlfwContext
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from collections import deque
from torchvision import transforms
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import CustomHopper
import os
import cv2

SEED = 42
GlfwContext(offscreen=True)

# Image preprocessing

def v_crop(pil_img, crop_top=0, crop_bottom=0):
    width, height = pil_img.size
    return pil_img.crop((0, crop_top, width, height - crop_bottom))

def h_crop(pil_img, crop_left=0, crop_right=0):
    width, height = pil_img.size
    return pil_img.crop((crop_left, 0, width - crop_right, height))

def isolate_and_grayscale(pil_img: Image.Image) -> Image.Image:
    image_np = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([30, 255, 200])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    result = cv2.bitwise_and(image_np, image_np, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(gray, mode='L')

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(lambda img: v_crop(img, crop_top=135, crop_bottom=65)),
    transforms.Lambda(lambda img: h_crop(img, crop_left=195, crop_right=175)),
    transforms.Lambda(lambda img: isolate_and_grayscale(img)),
    transforms.Resize((150, 65)),  # Resize mantenendo la proporzione
    transforms.ToTensor()
])

class CombinedWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        self.env = env

        # Initialize running stats for state normalization
        state_shape = env.observation_space.shape
        self.state_sum = np.zeros(state_shape)
        self.state_sumsq = np.zeros(state_shape)
        self.state_count = 1e-6  # avoid div by zero

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0.0, high=1.0, shape=(n_frames, 150, 65), dtype=np.float32),
            "state": env.observation_space
        })

    def update_state_stats(self, state):
        self.state_sum += state
        self.state_sumsq += np.square(state)
        self.state_count += 1

    def normalize_state(self, state):
        mean = self.state_sum / self.state_count
        var = (self.state_sumsq / self.state_count) - np.square(mean)
        std = np.sqrt(np.maximum(var, 1e-6))  # ensure no division by zero
        return (state - mean) / std

    def reset(self):
        state = self.env.reset()
        self.update_state_stats(state)
        norm_state = self.normalize_state(state)

        frame = self.env.render(mode='rgb_array')
        processed = preprocess(frame)
        for _ in range(self.n_frames):
            self.frames.append(processed.clone())
        return {
            "image": torch.cat(list(self.frames), dim=0).numpy(),
            "state": norm_state
        }

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.update_state_stats(state)
        norm_state = self.normalize_state(state)

        frame = self.env.render(mode='rgb_array')
        processed = preprocess(frame)
        self.frames.append(processed)
        return {
            "image": torch.cat(list(self.frames), dim=0).numpy(),
            "state": norm_state
        }, reward, done, info


class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4),
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

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(observation_space["state"].shape[0], 64),
            torch.nn.ReLU()
        )

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten + observation_space["state"].shape[0], features_dim),
            torch.nn.ReLU()
        )

    def forward(self, obs):
        image_feat = self.cnn(obs["image"])
        state_feat = obs["state"]
        return self.linear(torch.cat([image_feat, state_feat], dim=1))

def print_plot_rewards(rewards, title):
    x = np.arange(1, len(rewards)+1)
    plt.plot(x, rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

def main():
    task = "evaluate"  # or "evaluate"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_env = CombinedWrapper(Monitor(CustomHopper(domain='source')))
    test_env = CombinedWrapper(Monitor(CustomHopper(domain='target')))

    if task == "train":
        policy_kwargs = dict(
            features_extractor_class=CombinedExtractor,
            features_extractor_kwargs=dict(features_dim=512)
        )

        model = PPO("MultiInputPolicy", train_env, device='cuda', policy_kwargs=policy_kwargs, n_steps=1024, verbose=1)
        model.learn(total_timesteps=500_000)
        model.save("ppo_combined")

    elif task == "evaluate":
        model = PPO.load("ppo_combined", env=train_env, device='cuda')

    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, deterministic=True, render=False)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

if __name__ == '__main__':
    main()

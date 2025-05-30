# Estensore RL con frame differenziali e CNN semplice - Hopper + evaluate_policy
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
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

class FrameDiffWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_frame = None
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1, 84, 84), dtype=np.float32)

    def reset(self):
        self.env.reset()
        frame = self.env.render(mode='rgb_array')
        processed = preprocess(frame)
        self.prev_frame = processed
        zero_frame = torch.zeros_like(processed)
        return zero_frame.numpy()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        frame = self.env.render(mode='rgb_array')
        processed = preprocess(frame)
        diff = processed - self.prev_frame
        self.prev_frame = processed
        return diff.numpy(), reward, done, info

class SimpleCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
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
            torch.nn.Linear(n_flatten, features_dim),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.linear(self.cnn(x))

def print_plot_rewards(rewards, title):
    x = np.arange(1, len(rewards)+1)
    plt.plot(x, rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.xticks(x, labels=[str(val) for val in x])
    plt.show()
    with open("output_diff_frame_rewards.txt", "w") as file:
        for i in range(len(rewards)):
            file.write(f"Cumulative reward of episode {i+1}: {rewards[i]}\n")
        file.write(f"\nAverage return: {np.mean(rewards)}")

def main():
    task = "evaluate"  # oppure "train"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_env = FrameDiffWrapper(Monitor(CustomHopper(domain='source', param=None)))
    train_env.seed(SEED)
    train_env.action_space.seed(SEED)
    train_env.observation_space.seed(SEED)

    test_env = FrameDiffWrapper(Monitor(CustomHopper(domain='source', param=None)))
    test_env.seed(SEED)
    test_env.action_space.seed(SEED)
    test_env.observation_space.seed(SEED)

    print('State space:', train_env.observation_space)
    print('Action space:', train_env.action_space)
    print('Dynamics parameters:', train_env.unwrapped.get_parameters())

    if task == "train":
        policy_kwargs = dict(
            features_extractor_class=SimpleCNN,
            features_extractor_kwargs=dict(features_dim=512)
        )
        model = PPO("CnnPolicy", train_env, device='cuda', policy_kwargs=policy_kwargs,
                    n_steps=1028, clip_range=0.1, learning_rate=1e-3, verbose=1, seed=SEED)
        model.learn(total_timesteps=50_000)
        model.save("ppo_frame_diff_hopper")

    elif task == "evaluate":
        model = PPO.load("ppo_frame_diff_hopper", env=test_env, device='cuda')
        mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, return_episode_rewards=False)
        print(f"Mean reward over 50 episodes: {mean_reward} ± {std_reward}")

if __name__ == '__main__':
    main()

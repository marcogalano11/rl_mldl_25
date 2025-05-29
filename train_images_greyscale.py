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
from env.custom_hopper import CustomHopper
import os

SEED = 42

GlfwContext(offscreen=True)  # Create a window to init GLFW.

def v_crop(pil_img, crop_top=0, crop_bottom=0):
    width, height = pil_img.size
    top = crop_top
    bottom = height - crop_bottom
    return pil_img.crop((0, top, width, bottom))
  
def h_crop(pil_img, crop_left=0, crop_right=0):
    width, height = pil_img.size
    left = crop_left
    right = width - crop_right
    return pil_img.crop((left, 0, right, height))

import cv2
import numpy as np
from matplotlib import pyplot as plt

def isolate_and_grayscale(pil_img: Image.Image) -> Image.Image:
    # Converti PIL in NumPy RGB
    image_np = np.array(pil_img.convert("RGB"))  # shape: (H, W, 3)
    
    # Converti in HSV
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # Range per il marrone (può essere adattato)
    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([30, 255, 200])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Applica la maschera all'immagine RGB
    result = cv2.bitwise_and(image_np, image_np, mask=mask)

    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    # Converte di nuovo in immagine PIL (mode 'L' = grayscale)
    return Image.fromarray(gray, mode='L')

# Preprocessing immagini RGB
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((200, 200)),  # Resize to 200x200
    # transforms.CenterCrop((400,100)),
    transforms.Lambda(lambda img: v_crop(img, crop_top=135, crop_bottom=65)),
    transforms.Lambda(lambda img: h_crop(img, crop_left=195, crop_right=175)),  # Crop left 100 pixels
    transforms.Lambda(lambda img: isolate_and_grayscale(img)),
    transforms.ToTensor()  # shape: [C, H, W] ∈ [0,1]
])

# Stack RGB 
class RGBStackWrapper(gym.ObservationWrapper):

    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0,
            shape=(n_frames, 300, 130),
            dtype=np.float32
        )

    def reset(self):
        self.env.reset()
        frame = self.env.render(mode='rgb_array')
        processed = preprocess(frame)

        # salva immagine 
        pil_image = transforms.ToPILImage()(processed)
        os.makedirs('frames', exist_ok=True)
        pil_image.save('frames/frame_reset.png')

        for _ in range(self.n_frames):
            self.frames.append(processed.clone())
        return torch.cat(list(self.frames), dim=0).numpy()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        frame = self.env.render(mode='rgb_array')
        processed = preprocess(frame)
        self.frames.append(processed)
        return torch.cat(list(self.frames), dim=0).numpy(), reward, done, info

# CNN
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4),
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

# Plot function
def print_plot_rewards(rewards, title):
    x = np.arange(1, len(rewards)+1)
    plt.plot(x, rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.xticks(x, labels=[str(val) for val in x])
    plt.show()

    with open("output_greyscale_source_source.txt", "w") as file:
        for i in range(len(rewards)):
            file.write(f"Cumulative reward of episode {i+1}: {rewards[i]}\n")
        file.write(f"\nAverage return: {np.mean(rewards)}")


def main():

    task = "evaluate" 
    #task = "train" 

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_env = RGBStackWrapper(Monitor(CustomHopper(domain='source', param=None)))
    train_env.seed(SEED)
    train_env.action_space.seed(SEED)
    train_env.observation_space.seed(SEED)
    
    test_env = RGBStackWrapper(Monitor(CustomHopper(domain='source', param=None)))
    test_env.seed(SEED)
    test_env.action_space.seed(SEED)
    test_env.observation_space.seed(SEED)

    print('State space:', train_env.observation_space)
    print('Action space:', train_env.action_space)
    print('Dynamics parameters:', train_env.unwrapped.get_parameters())

    if task=="train":

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512)
        )

        model = PPO("CnnPolicy", train_env, device='cuda', policy_kwargs=policy_kwargs, n_steps=1028, clip_range=0.1, learning_rate=1e-3, verbose=1, seed=SEED)
        model.learn(total_timesteps=50_000)
        model.save("ppo_rgb_4frame_source")
    
    elif task=="evaluate":

        model = PPO.load("ppo_rgb_4frame_source", env=train_env, device='cuda')

    # Evaluation 
    num_episodes = 50
    rewards = np.zeros(num_episodes)
    i = 0
    obs = test_env.reset()
    print("Observation shape:", obs.shape)

    cumulative_reward = 0

    while i < num_episodes:
        action, _states = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        cumulative_reward += reward
        if done:
            rewards[i] = cumulative_reward
            cumulative_reward = 0
            obs = test_env.reset()
            i += 1

    print(f"Mean reward over 50 episodes: {np.mean(rewards)}")
    print_plot_rewards(rewards, "Evaluation PPO + 4 RGB frames (Source)")

if __name__ == '__main__':
    main()

import os
import gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from gym import spaces
from PIL import Image
from torchvision import transforms
import cv2
from torch.utils.data import Dataset, DataLoader
from env.custom_hopper import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from mujoco_py import GlfwContext
import random
from images.image_preprocessing import preprocess

GlfwContext(offscreen=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class StateOnlyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = env.observation_space

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state, reward, done, info
    
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
    

class ImageOnlyExtractor(nn.Module):
    def __init__(self, n_frames=8, features_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_frames, 84, 84)
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 3)
        )

    def forward(self, x):
        features = self.cnn(x)
        lin_features = self.linear(features)
        #print("\n\nImage actions:", lin_features)
        os.makedirs("logs", exist_ok=True)
        with open("logs/image_actions.csv", "a") as f:
            np.savetxt(f, lin_features.detach().cpu().numpy(), delimiter=",")
        return lin_features

def generate_teacher_dataset(teacher_model, env_state, env_image, output_path="teacher_dataset.npz", num_episodes=50):
    import os
    import numpy as np

    obs_image_list = []
    action_list = []

    for ep in range(num_episodes):
        obs_state = env_state.reset()
        obs_image = env_image.reset()

        done_state = False
        done_image = False

        while not (done_state or done_image):
            action, _ = teacher_model.predict(obs_state, deterministic=True)

            obs_image_list.append(obs_image)
            action_list.append(action)

            try:
                obs_state, _, done_state, _ = env_state.step(action)
            except RuntimeError:
                print("[!] env_state requires reset")
                break

            try:
                obs_image, _, done_image, _ = env_image.step(action)
            except RuntimeError:
                print("[!] env_image requires reset")
                break

    """print("\n\nImages start:", obs_image_list[:10])
    print("\n\nActions:", action_list[:10])
    print("\n\nImages end:", obs_image_list[len(obs_image_list)-10:])
    print("\n\nActions:", action_list[len(obs_image_list)-10:])
"""
    obs_image_array = np.stack(obs_image_list)
    action_array = np.stack(action_list)

    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    np.savez_compressed(output_path, images=obs_image_array, actions=action_array)
    print(f"[✓] Dataset saved: {output_path}")
    

class TeacherDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.images = torch.tensor(data["images"], dtype=torch.float32)
        self.actions = torch.tensor(data["actions"], dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.actions[idx]
    

def train_student(model, dataloader, epochs=10, lr=1e-4, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, actions in dataloader:
            images = images.to(device)
            actions = actions.to(device)

            preds = model(images)
            loss = criterion(preds, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


def evaluate_policy(model, env, n_episodes=50, is_torch_model=False, device='cpu'):
    returns = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            if is_torch_model:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(obs_tensor).cpu().numpy()[0]
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, _ = env.step(action)
            total_reward += reward

        returns.append(total_reward)

    avg_reward = np.mean(returns)
    std_reward = np.std(returns)
    print(f"[✓] Evaluation - Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    return avg_reward

def main():
    # 1. Teacher
    train_env_state = StateOnlyWrapper(Monitor(CustomHopper(domain='source')))
    train_env_image = ImageOnlyWrapper(Monitor(CustomHopper(domain='source')))

    test_env_state = StateOnlyWrapper(Monitor(CustomHopper(domain='target')))
    test_env_image = ImageOnlyWrapper(Monitor(CustomHopper(domain='target')))

    teacher_model = PPO.load("rl_mldl_25/ppo/tuned_ppo_source", env=train_env_state)

    # 2. Dataset
    print("Generating dataset from teacher on SOURCE domain...")
    generate_teacher_dataset(
        teacher_model=teacher_model,
        env_state=train_env_state,
        env_image=train_env_image,
        output_path="teacher_dataset.npz",
        num_episodes=50
    )
    
    # 3. Student
    dataset = TeacherDataset("teacher_dataset.npz")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    student_model = ImageOnlyExtractor()
    train_student(student_model, dataloader, epochs=10)

    # 4. Save
    torch.save(student_model.state_dict(), "student_policy.pt")
    print("Student model saved.")

    # 5. Evaluation
    print("\n[Evaluation] Teacher policy on SOURCE domain:")
    evaluate_policy(teacher_model, train_env_state)

    print("\n[Evaluation] Student policy on SOURCE domain:")
    student_model.eval()
    evaluate_policy(student_model, train_env_image, is_torch_model=True, device='cpu')

    print("\n[Evaluation] Teacher policy on TARGET domain:")
    evaluate_policy(teacher_model, test_env_state)

    print("\n[Evaluation] Student policy on TARGET domain:")
    student_model.eval()
    evaluate_policy(student_model, test_env_image, is_torch_model=True, device='cpu')

if __name__ == "__main__":
    main()
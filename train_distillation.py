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
from torch.utils.data import Dataset
import glob
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from gym import spaces
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

GlfwContext(offscreen=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class RewardLoggerCallback(BaseCallback):
    def __init__(self, save_path='rl_rewards.npy', verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.save_path = save_path

    def _on_step(self) -> bool:
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]
            if 'episode' in info:
                ep_reward = info['episode']['r']
                self.rewards.append(ep_reward)
        return True

    def _on_training_end(self) -> None:
        np.save(self.save_path, np.array(self.rewards))
        if self.verbose:
            print(f"[✓] Saved RL training rewards to {self.save_path}")

def plot_rl_rewards(reward_file: str = "rl_distillation_rewards.npy", save_path: str = None):

    if not os.path.exists(reward_file):
        print(f"[!] Il file '{reward_file}' non esiste.")
        return

    rewards = np.load(reward_file)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Reward per episode')
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.title("Reward during RL training")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"[✓] Plot saved in {save_path}")
    else:
        plt.show()

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
    transforms.Lambda(lambda img: v_crop(img, crop_top=125, crop_bottom=55)),
    transforms.Lambda(lambda img: h_crop(img, crop_left=90, crop_right=90)),
    transforms.Lambda(lambda img: isolate_and_grayscale(img)),
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

class ImageOnlyWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_frames=8):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        self.env = env

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_frames, 64, 64), dtype=np.float32
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
    def __init__(self, n_frames=8, features_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_frames, 64, 64)
            n_flatten = self.cnn(dummy).shape[1]

        self.output_dim = features_dim
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        features = self.cnn(x)
        return self.linear(features)

class SupervisedPolicy(nn.Module):
    def __init__(self, feature_extractor: ImageOnlyExtractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.mlp_head = nn.Sequential(
            nn.Linear(feature_extractor.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.mlp_head(features)

def generate_teacher_dataset_to_disk(teacher_model, env_state, env_image, output_dir="dataset_eps", num_episodes=500):
    os.makedirs(output_dir, exist_ok=True)

    for ep in range(num_episodes):
        obs_state = env_state.reset()
        obs_image = env_image.reset()

        images = []
        actions = []

        done_state = False
        done_image = False

        while not (done_state or done_image):
            action, _ = teacher_model.predict(obs_state, deterministic=True)

            images.append(obs_image.astype(np.float16))
            actions.append(action.astype(np.float32))

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

        episode_path = os.path.join(output_dir, f"ep_{ep:04d}.npz")
        np.savez_compressed(episode_path,
                            images=np.stack(images),
                            actions=np.stack(actions))
        print(f"[✓] Saved episode {ep+1}/{num_episodes} to {episode_path}")
    
    print("[✓] All episodes saved.")

class TeacherDiskDataset(Dataset):
    def __init__(self, folder_path):
        self.file_paths = sorted(glob.glob(os.path.join(folder_path, "*.npz")))
        self.samples = []

        for path in self.file_paths:
            data = np.load(path)
            n = len(data['images'])
            self.samples.extend([(path, i) for i in range(n)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, i = self.samples[idx]
        data = np.load(file_path)
        image = torch.tensor(data['images'][i], dtype=torch.float32)
        action = torch.tensor(data['actions'][i], dtype=torch.float32)
        return image, action
    

def train_student(policy_model, dataloader, epochs=10, lr=1e-4, device='cuda'):
    policy_model.to(device)
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        policy_model.train()
        total_loss = 0

        for i, (images, actions) in enumerate(dataloader):
            images = images.to(device)
            actions = actions.to(device)

            preds = policy_model(images)
            loss = criterion(preds, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"[✓] Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

def evaluate_policy(model, env, n_episodes=50, is_torch_model=False, device='cpu', max_steps_per_episode=500):
    returns = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done and step_count < max_steps_per_episode:
            if is_torch_model:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(obs_tensor).cpu().numpy()[0]
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, _ = env.step(action)
            total_reward += reward
            step_count += 1

            

        returns.append(total_reward)

    avg_reward = np.mean(returns)
    std_reward = np.std(returns)
    print(f"[✓] Evaluation - Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    return avg_reward

class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.extractor = ImageOnlyExtractor(n_frames=8, features_dim=features_dim)

    def forward(self, observations):
        return self.extractor(observations)

class CustomCNNPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            **kwargs
        )

def train_student_with_rl(student_model, steps=1_000_000):
    env = ImageOnlyWrapper(Monitor(CustomHopper(domain='source')))

    print("[✓] Fine-tuning student model via RL...")
    model = PPO(CustomCNNPolicy, env, verbose=1, device='cuda')

    # Copia i pesi dal supervised model nel feature extractor della policy
    if student_model is not None:
        model.policy.features_extractor.extractor.load_state_dict(
            student_model.feature_extractor.state_dict()
        )
        for param in model.policy.features_extractor.extractor.parameters():
            param.requires_grad = False
        print("[✓] Loaded weights from supervised model into RL policy.")

    reward_callback = RewardLoggerCallback(save_path='rl_distillation_rewards.npy', verbose=1)

    model.learn(total_timesteps=steps, callback=reward_callback)

    return model

def main(generate_dataset):
    num_episodes = 1000        
    num_epochs = 20          

    dataset_name = f"teacher_dataset_{num_episodes}eps"
    student_policy_name = f"student_policy_{num_episodes}eps_{num_epochs}epochs.pt"
    rl_model_name = f"student_rl_finetuned_{num_episodes}eps_{num_epochs}epochs"

    # 1. Teacher
    train_env_state = Monitor(CustomHopper(domain='source'))
    train_env_image = ImageOnlyWrapper(Monitor(CustomHopper(domain='source')))
    test_env_image = ImageOnlyWrapper(Monitor(CustomHopper(domain='target')))

    teacher_model = PPO.load("ppo/tuned_ppo", env=train_env_state)

    # 2. Dataset
    if generate_dataset:
        print(f"Generating dataset ({num_episodes} episodes)...")
        generate_teacher_dataset_to_disk(
            teacher_model=teacher_model,
            env_state=train_env_state,
            env_image=train_env_image,
            output_dir=dataset_name,
            num_episodes=num_episodes
        )
    else:
        print(f"[!] Skipping dataset generation. Using: {dataset_name}")

    # 3. Student
    dataset = TeacherDiskDataset(dataset_name)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    if os.path.exists(student_policy_name) :
        print(f"[✓] Loading existing student policy and extractor...")
        student_policy = torch.load(student_policy_name) 
        student_policy.eval()
    else:
        print("[✓] Starting supervised training...")
        extractor = ImageOnlyExtractor()
        student_policy = SupervisedPolicy(extractor)
        train_student(student_policy, dataloader, epochs=num_epochs)
        torch.save(student_policy, student_policy_name)
        print("Student model saved.")


    # 4. RL fine-tuning
    if os.path.exists(f"{rl_model_name}.zip"):
        print(f"[✓] RL fine-tuned model found at {rl_model_name}.zip — skipping RL training.")
        rl_model = PPO.load(rl_model_name, env=train_env_image)
    else:
        rl_model = train_student_with_rl(student_model=student_policy)
        rl_model.save(rl_model_name)
        print(f"[✓] RL fine-tuned model saved to {rl_model_name}")
        plot_rl_rewards(save_path="rl_distillation_plot.png")

    # 5. Evaluation

    print("\n[Evaluation] Supervised policy on SOURCE domain:")
    evaluate_policy(student_policy, train_env_image, is_torch_model=True, device="cuda")

    print("\n[Evaluation] Fine_tuned policy on SOURCE domain:")
    evaluate_policy(rl_model, train_env_image)

    print("\n[Evaluation] Supervised policy on TARGET domain:")
    evaluate_policy(student_policy, test_env_image, is_torch_model=True, device="cuda")

    print("\n[Evaluation] Fine_tuned policy on TARGET domain:")
    evaluate_policy(rl_model, test_env_image)

if __name__ == "__main__":
    main(generate_dataset=False)
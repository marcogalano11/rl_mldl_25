from mujoco_py import GlfwContext
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import CustomHopper
from images.images_only_env import ImageOnlyExtractor, ImageOnlyWrapper

SEED = 42
GlfwContext(offscreen=True)

def print_plot_rewards(rewards, title):
    x = np.arange(1, len(rewards)+1)
    plt.plot(x, rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

def main():
    task = "train"  # "train" or "evaluate"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_env = ImageOnlyWrapper(Monitor(CustomHopper(domain='source')))
    test_env = ImageOnlyWrapper(Monitor(CustomHopper(domain='source')))

    # Training
    if task == "train":
        policy_kwargs = dict(
            features_extractor_class=ImageOnlyExtractor,
            features_extractor_kwargs=dict(features_dim=512) 
        )

        model = PPO("CnnPolicy", train_env, device='cpu', policy_kwargs=policy_kwargs, n_steps=1024, verbose=1)
        model.learn(total_timesteps=1_000_000)
        model.save("images/ppo_img_only")

    elif task == "evaluate":
        model = PPO.load("images/ppo_img_only", env=train_env, device='cpu')

    # Evaluation
    list_rewards, _ = evaluate_policy(model, test_env, n_eval_episodes=50, deterministic=True, render=False, return_episode_rewards=True)
    mean_reward = np.mean(list_rewards)
    std_reward = np.std(list_rewards)
    print(f"\nMean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print_plot_rewards(list_rewards, title="Rewards PPO on images")

if __name__ == '__main__':
    main()
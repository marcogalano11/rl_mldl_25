import numpy as np
import torch
import matplotlib.pyplot as plt
import os

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

def plot_rl_rewards(reward_file: str = "distillation/outputs/rl_distillation_rewards.npy", save_path: str = None):

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
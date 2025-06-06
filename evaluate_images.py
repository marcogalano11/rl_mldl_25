import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from train_images_states_nonorm import CombinedWrapper

def evaluate(model, test_env, n_eval_episodes):
    list_rewards, _ = evaluate_policy(model, test_env, n_eval_episodes=n_eval_episodes, deterministic=True, render=False, return_episode_rewards=True)

    # Mostra reward per episodio
    for i, reward in enumerate(list_rewards, 1):
        print(f"Episode {i}: Reward = {reward:.2f}")

    # Media e deviazione standard
    mean_reward = np.mean(list_rewards)
    std_reward = np.std(list_rewards)
    print(f"\nMean reward: {mean_reward:.2f} ± {std_reward:.2f}")


if __name__ == "__main__":
    # Load your environment and trained model
    from env.custom_hopper import CustomHopper
    from stable_baselines3 import PPO

    modes = ["both", "images", "states"]
    
    for mode in modes:
        print(f"\n--- Evaluating mode: {mode} ---")
        test_env = CombinedWrapper(CustomHopper(domain='target'), evaluation=mode)
        model = PPO.load("ppo_combined", env=test_env)
        evaluate(model, test_env, n_eval_episodes=10)
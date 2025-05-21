import gym
import argparse
import numpy as np
import itertools
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import random
import torch
import csv

SEED = 42
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_env(domain="target"):
    env = gym.make(f'CustomHopper-{domain}-v0')
    env.seed(SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)  
    return env

def train_and_evaluate(args):

    set_global_seed(SEED)

    learning_rates = [3e-4, 1e-3]
    clip_ranges = [0.1, 0.2, 0.3]
    n_steps_list = [1024, 2048, 4096]

    test_env = make_env(domain=args.test_domain)

    results = []

    for lr, clip, n_steps in itertools.product(learning_rates, clip_ranges, n_steps_list):
        print(f"\n🔍 Testing: LR={lr}, Clip={clip}, N_STEPS={n_steps}")
        model_name = f"ppo_{args.train_domain}_lr{lr}_clip{clip}_nsteps{n_steps}"
        train_env = make_env(domain=args.train_domain)

        if args.train:
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=lr,
                clip_range=clip,
                n_steps=n_steps,
                verbose=0,
                seed=SEED
            )
            model.learn(total_timesteps=args.timesteps)
            model.save(model_name)
        else:
            model = PPO.load(model_name)

        # EVALUATION
        mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50)
        print(f"✅ Reward: {mean_reward:.2f} ± {std_reward:.2f}")

        results.append({
            "learning_rate": lr,
            "clip_range": clip,
            "n_steps": n_steps,
            "mean_reward": mean_reward,
            "std_reward": std_reward
        })


    # Ordina e stampa le top 3 configurazioni
    sorted_results = sorted(results, key=lambda x: x["mean_reward"], reverse=True)

    csv_filename = "ppo_grid_search_results.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted_results[0].keys())
        writer.writeheader()
        writer.writerows(sorted_results)

    print(f"\n📁 Risultati salvati in: {csv_filename}")
    print("\n🏆 Top 3 Configurazioni:")
    for r in sorted_results[:3]:
        print(f"→ LR: {r['learning_rate']}, Clip: {r['clip_range']}, n_steps: {r['n_steps']} → Reward: {r['mean_reward']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search PPO on CustomHopper")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--train_domain", type=str, default="source", choices=["source", "target"], help="Env domain for training")
    parser.add_argument("--test_domain", type=str, default="target", choices=["source", "target"], help="Env domain for testing")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Number of training timesteps")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes per config")
    args = parser.parse_args()

    train_and_evaluate(args)
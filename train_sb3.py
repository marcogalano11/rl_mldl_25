"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import itertools
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import argparse
import os

SEED = 42

folder = "ppo/outputs"
os.makedirs(folder, exist_ok=True)

def main():

    args = parse_args()

    train_env = Monitor(gym.make('CustomHopper-source-v0'))
    train_env.seed(SEED)
    train_env.action_space.seed(SEED)
    train_env.observation_space.seed(SEED)

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    policy = args.policy
    task = args.task

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    if policy=="PPO":

        if task=="tuning":

            n_steps_list = [2048, 4096]
            learning_rates = [3e-4, 1e-3]
            clip_ranges = [0.1, 0.2, 0.3]


            for n_steps, lr, clip_range in itertools.product(n_steps_list, learning_rates, clip_ranges):

                config_name = f"ppo_nsteps_{n_steps}_lr_{lr}_cliprange_{clip_range}"

                train_env = Monitor(gym.make('CustomHopper-source-v0'))
                train_env.seed(SEED)
                train_env.action_space.seed(SEED)
                train_env.observation_space.seed(SEED)


                model = PPO("MlpPolicy", train_env, verbose=0, n_steps=n_steps, learning_rate=lr, clip_range=clip_range, seed=SEED)

                print(f"Training: {config_name}")

                model.learn(total_timesteps=1e6)

                test_env = Monitor(gym.make('CustomHopper-source-v0'))
                test_env.seed(SEED)
                test_env.action_space.seed(SEED)
                test_env.observation_space.seed(SEED)

                mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, deterministic=True, render=False)

                with open(f"{folder}/tuning_results.txt", "a") as tuning_results:
                    tuning_results.write(f"{config_name}; avg: {mean_reward}, std: {std_reward}\n")

        elif task=="bounding":

            params = {"n_steps": 4096, "lr": 3e-4, "clip_range": 0.1}

            print("Evaluating Lower Bound\n")
            evaluate_bounds(params, environment="source")
            print("Evaluating Upper Bound\n")
            evaluate_bounds(params, environment="target")

        else:
            model = PPO("MlpPolicy", train_env, verbose=1, seed=SEED)

            model.learn(total_timesteps=1e6)

            model.save("ppo_hopper")

            evaluate(model, train_env)

    else: #SAC policy, not tuned nor bounded
        model = SAC("MlpPolicy", train_env, verbose=1)

        model.learn(total_timesteps=500_000, log_interval=4)

        model.save("sac_hopper")

        evaluate(model, train_env)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', choices=['PPO', 'SAC'], default='PPO')
    parser.add_argument('--task', choices=['tuning', 'bounding'], default=None)
    return parser.parse_args()

def evaluate(model, test_env):
    
    obs = test_env.reset()
    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, deterministic=True, render=False)
    print(f"Average reward: {mean_reward}, Average std: {std_reward}")

def evaluate_bounds(parameters, environment="source"):

    train_env = Monitor(gym.make(f"CustomHopper-{environment}-v0"))
    train_env.seed(SEED)
    train_env.action_space.seed(SEED)
    train_env.observation_space.seed(SEED)

    model = PPO("MlpPolicy", train_env, verbose=1, n_steps=parameters["n_steps"], 
                learning_rate=parameters["lr"], clip_range=parameters["clip_range"], seed=SEED)
    model.learn(total_timesteps=1e6)
    model.save(f"tuned_ppo_{environment}")

    means = np.zeros(3)
    stds = np.zeros(3)

    for i in range(3):

        test_env = Monitor(gym.make('CustomHopper-target-v0'))
        test_env.seed(SEED)
        test_env.action_space.seed(SEED)
        test_env.observation_space.seed(SEED)

        mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, deterministic=True, render=False)
        means[i] = mean_reward
        stds[i] = std_reward

    filepath = f"{folder}/bounds.txt"
            
    with open(filepath, "a") as bounds:
        bounds.write(f"Bound {environment}-target: Average mean: {means.mean()}, Average std: {stds.mean()}\n")

def print_plot_rewards(rewards,title):
    x = np.arange(1,len(rewards)+1)
    plt.plot(x, rewards)
    plt.title(title)
    plt.xticks(x, labels=[str(val) for val in x])
    plt.show()

    print(f"Printing cumulative rewards of {title}\n")
    for i in range(len(rewards)):
        print(f"Cumulative reward of episode {i+1}: {rewards[i]}\n")
    print(f"\nAverage cumulative reward: {np.mean(rewards)}\n")
    
if __name__ == '__main__':
    main()
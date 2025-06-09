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
from train_sb3 import evaluate
import os

SEED = 42

folder = "ppo/outputs"
os.makedirs(folder, exist_ok=True)

def main():

    args = parse_args()

    tuning = args.tuning

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    params = {"n_steps": 4096, "lr": 3e-4, "clip_range": 0.1}

    train_env = Monitor(gym.make('CustomHopper-source-v0', param=0.1, distribution="uniform"))
    train_env.seed(SEED)
    train_env.action_space.seed(SEED)
    train_env.observation_space.seed(SEED)
    
    test_env = Monitor(gym.make('CustomHopper-target-v0'))
    test_env.seed(SEED)
    test_env.action_space.seed(SEED)
    test_env.observation_space.seed(SEED)

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    if tuning:

        distributions = ["uniform", "normal", "lognormal"]
        distr_params = [0.1,0.4,0.7,1]


        for distribution, param in itertools.product(distributions, distr_params):
            
            config_name = f"randomized_ppo_distribution_{distribution}_{param}"

            train_env = Monitor(gym.make('CustomHopper-source-v0', param=param, distribution=distribution))
            train_env.seed(SEED)
            train_env.action_space.seed(SEED)
            train_env.observation_space.seed(SEED)


            model = PPO("MlpPolicy", train_env, verbose=0, n_steps=params["n_steps"], 
                        learning_rate=params["lr"], clip_range=params["clip_range"], seed=SEED)

            print(f"Training: {config_name}")

            model.learn(total_timesteps=1e6)

            test_env = Monitor(gym.make('CustomHopper-source-v0'))
            test_env.seed(SEED)
            test_env.action_space.seed(SEED)
            test_env.observation_space.seed(SEED)

            mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, deterministic=True, render=False)

            with open(f"{folder}/tuning_results_randomization.txt", "a") as tuning_results:
                tuning_results.write(f"{config_name}; avg: {mean_reward}, std: {std_reward}\n")

    else:

        distribution = {"distr":"lognormal", "param":0.1}

        train_env = Monitor(gym.make('CustomHopper-source-v0', param=distribution["param"], 
                                     distribution=distribution["distr"]))
        train_env.seed(SEED)
        train_env.action_space.seed(SEED)
        train_env.observation_space.seed(SEED)
        
        test_env = Monitor(gym.make('CustomHopper-target-v0'))
        test_env.seed(SEED)
        test_env.action_space.seed(SEED)
        test_env.observation_space.seed(SEED)

        model = PPO("MlpPolicy", train_env, verbose=1, n_steps=params["n_steps"], 
                    learning_rate=params["lr"], clip_range=params["clip_range"], seed=SEED)
        model.learn(total_timesteps=1e6)
        model.save("randomized_ppo")

        evaluate(model, test_env)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tuning', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    main()
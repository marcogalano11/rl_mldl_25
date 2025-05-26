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

SEED = 42

def main():
    # to be changed when shifting the training and testing environment
    train_env = Monitor(gym.make('CustomHopper-source-v0'))
    test_env = Monitor(gym.make('CustomHopper-target-v0'))

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    policy = "PPO"
    tuning = True
    training = False
    evaluation = False

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    if policy=="PPO":

        if tuning:

            n_steps_list = [1024] #[1024, 2048, 4096]
            learning_rates = [1e-5, 3e-4, 1e-3]
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

                test_env = Monitor(gym.make('CustomHopper-target-v0'))
                test_env.seed(SEED)
                test_env.action_space.seed(SEED)
                test_env.observation_space.seed(SEED)

                mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, deterministic=True, render=False)

                with open("tuning_results_marco.txt", "a") as tuning_results:
                    tuning_results.write(f"{config_name}; avg: {mean_reward}, std: {std_reward}\n")

        elif training:
            print("Evaluating Lower Bound\n")
            evaluate_bounds("source")
            print("Evaluating Upper Bound")
            evaluate_bounds("target")

    else:
        if training:

            model = SAC("MlpPolicy", train_env, verbose=1)

            model.learn(total_timesteps=500_000, log_interval=4)

            model.save("sac_hopper_target")

        else:
            # del model #this only if we have trained a model in this script and we want to delete it
            model = SAC.load("sac_hopper") #sac_hopper for the source environment else sac_hopper_target

    #EVALUATION
    if evaluation:
        obs = test_env.reset()

        cumulative_reward = 0
        i = 0
        num_episodes = 50
        rewards = np.zeros(num_episodes)
                
        while i < num_episodes:
            action, _states = model.predict(obs)
            obs, reward, done, info = test_env.step(action)
            cumulative_reward += reward
            #test_env.render()
            if done:
                #print(f"Cumulative reward of episode {i+1}: {cumulative_reward}")
                rewards[i] = cumulative_reward
                cumulative_reward = 0
                obs = test_env.reset()
                i += 1
        title = "Simulation on a Source-Target environment with SAC" #change when evaluating
        print_plot_rewards(rewards,title)


def evaluate_bounds(environment="source"):
    train_env = Monitor(gym.make(f"CustomHopper-{environment}-v0"))
    train_env.seed(SEED)
    train_env.seed(SEED)
    train_env.action_space.seed(SEED)
    train_env.observation_space.seed(SEED)

    model = PPO("MlpPolicy", train_env, verbose=1, n_steps=4096, learning_rate=3e-4, clip_range=0.2, seed=SEED)
    model.learn(total_timesteps=1e6)
    model.save(f"tuned_ppo_{environment}")
    #model = PPO.load(f"tuned_ppo_{environment}")
    means = np.zeros(3)
    stds = np.zeros(3)

    for i in range(3):

        test_env = Monitor(gym.make('CustomHopper-target-v0'))
        test_env.seed(SEED)
        test_env.seed(SEED)
        test_env.action_space.seed(SEED)
        test_env.observation_space.seed(SEED)

        mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, deterministic=True, render=False)
        means[i] = mean_reward
        stds[i] = std_reward

    #print(f"Bound {environment}-target: Average mean: {means.mean()}, Average std: {stds.mean()}\n")
            
    with open("bounds.txt", "a") as bounds:
        bounds.write(f"Bound {environment}-target: Average mean: {means.mean()}, Average std: {stds.mean()}\n")

def print_plot_rewards(rewards,title):
    x = np.arange(1,len(rewards)+1)
    plt.plot(x, rewards)
    plt.title(title)
    plt.xticks(x, labels=[str(val) for val in x])
    plt.show()

    with open("output_source_target_sac.txt", "w") as file: #change the name of the file when evaluating
        for i in range(len(rewards)):
            file.write(f"Cumulative reward of episode {i+1}: {rewards[i]}\n")
        file.write(f"\nAverage return: {np.mean(rewards)}")
    
    

if __name__ == '__main__':
    main()
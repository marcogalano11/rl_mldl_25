"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO, SAC
import numpy as np
import matplotlib.pyplot as plt

def main():
    # to be changed when shifting the training and testing environment
    train_env = gym.make('CustomHopper-target-v0')
    test_env = gym.make('CustomHopper-target-v0')

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    ppo_policy = False
    training = False

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    if ppo_policy:

        if training:
            model = PPO("MlpPolicy", train_env, verbose=1)

            model.learn(total_timesteps=1e6)

            model.save("ppo_hopper_target")
        
        else:
            # del model #this only if we have trained a model in this script and we want to delete it
            model = PPO.load("ppo_hopper") #ppo_hopper for the source environment else ppo_hopper_target

    else:
        if training:
            model = SAC("MlpPolicy", train_env, verbose=1)

            model.learn(total_timesteps=500_000, log_interval=4)

            model.save("sac_hopper_target")

        else:
            # del model #this only if we have trained a model in this script and we want to delete it
            model = SAC.load("sac_hopper") #sac_hopper for the source environment else sac_hopper_target

    #EVALUATION
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
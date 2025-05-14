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

    """
    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper
    """
    #ppo_policy = False
    training = True
    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #
    modelli= ['PPO','SAC']
    params = [0.1, 0.3, 0.5, 0.7, 1.0]
    for modello in modelli:
        for param in params:
            print(f"\nTraining with param={param}")

            train_env = CustomHopper(domain='source', param=param)
            test_env = CustomHopper(domain='source', param=None)

            if modello=='PPO':

                if training:
                    model = PPO("MlpPolicy", train_env)

                    model.learn(total_timesteps=1_000_000)

                    model.save(f"ppo_hopper_target_param_{param}")

                
                else:
                    # del model #this only if we have trained a model in this script and we want to delete it
                    model = PPO.load("ppo_hopper") #ppo_hopper for the source environment else ppo_hopper_target

            elif modello == 'SAC':
                if training:
                    model = SAC("MlpPolicy", train_env)

                    model.learn(total_timesteps=1_000_000, log_interval=4)

                    model.save(f"sac_hopper_target_param_{param}")

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
            #title = f"Simulation on a Source-Target environment with {model}" #change when evaluating
            with open(f"output_source_target_{modello}_UDR_param_{param}.txt", "w") as file: #change the name of the file when evaluating
                for i in range(len(rewards)):
                    file.write(f"Cumulative reward of episode {i+1}: {rewards[i]}\n")
                file.write(f"\nAverage return: {np.mean(rewards)}")
            

if __name__ == '__main__':
    main()
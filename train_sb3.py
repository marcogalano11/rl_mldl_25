"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO, SAC

def main():
    train_env = gym.make('CustomHopper-source-v0')

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    ppo_policy = False

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    if ppo_policy:

        model = PPO("MlpPolicy", train_env, verbose=1)

        model.learn(total_timesteps=1e6)

        model.save("ppo_hopper")

        #If we want to use deletion and reloading
        """
            # del model #this only if we have trained a model in this script and we want to delete it
            model = PPO.load("ppo_hopper")
        """
    else:

        model = SAC("MlpPolicy", train_env, verbose=1)

        model.learn(total_timesteps=500_000, log_interval=4)

        model.save("sac_hopper")

        #If we want to use deletion and reloading
        """
            # del model #this only if we have trained a model in this script and we want to delete it
            model = SAC.load("sac_hopper")
        """

    #EVALUATION
    obs = train_env.reset()

    cumulative_reward = 0
    i = 0
            
    while i <10:
        action, _states = model.predict(obs)
        obs, reward, done, info = train_env.step(action)
        cumulative_reward += reward
        #train_env.render()
        if done: 
            i += 1
            print(f"Cumulative reward of episode {i}: {cumulative_reward}")
            cumulative_reward = 0
            obs = train_env.reset()
        


if __name__ == '__main__':
    main()
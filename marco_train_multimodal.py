import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Import your env and extractor from the module file, e.g.:
from env.custom_hopper_multimodal import CustomHopperMultimodal, CustomCombinedExtractor

def train():
    
    env = make_vec_env("CustomHopperMultimodal-source-v0", n_envs=1)

    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor)

    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    model.learn(total_timesteps=200000)

    model.save("ppo_customhopper_multimodal")

if __name__ == "__main__":
    train()

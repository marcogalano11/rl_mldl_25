import gym
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import *
from stable_baselines3.common.monitor import Monitor

def main(domain="source"):
   
    model_path = "ppo/tuned_ppo_source" 
    teacher_model = PPO.load(model_path)

    env_name = f"CustomHopper-{domain}-v0"
    env = Monitor(gym.make(env_name))

    mean_reward, std_reward = evaluate_policy(teacher_model, env, n_eval_episodes=20, deterministic=True)
    print(f"[✓] Teacher evaluation on {domain.upper()} - Reward: {mean_reward:.2f} ± {std_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, choices=["source", "target"], required=True,
                        help="Scegli il dominio: source o target")

    args = parser.parse_args()
    main(args.domain)
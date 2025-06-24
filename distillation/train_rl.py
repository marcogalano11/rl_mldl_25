from models import CustomCNNPolicy
from wrapper import ImageOnlyWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from env.custom_hopper import CustomHopper

class RewardLoggerCallback(BaseCallback):
    def __init__(self, save_path='rl_rewards.npy', verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.save_path = save_path

    def _on_step(self) -> bool:
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]
            if 'episode' in info:
                ep_reward = info['episode']['r']
                self.rewards.append(ep_reward)
        return True

    def _on_training_end(self) -> None:
        np.save(self.save_path, np.array(self.rewards))
        if self.verbose:
            print(f"Saved RL training rewards to {self.save_path}")

def train_student_with_rl(student_model, steps=1_000_000):
    env = ImageOnlyWrapper(Monitor(CustomHopper(domain='source')))

    print("Fine-tuning student model via RL..")
    model = PPO(CustomCNNPolicy, env, verbose=1, device='cuda')

    # Copia i pesi dal supervised model nel feature extractor della policy
    if student_model is not None:
        model.policy.features_extractor.extractor.load_state_dict(
            student_model.feature_extractor.state_dict()
        )
        for param in model.policy.features_extractor.extractor.parameters():
            param.requires_grad = False
        print("Loaded weights from supervised model into RL policy.")

    reward_callback = RewardLoggerCallback(save_path='distillation/outputs/rl_distillation_rewards.npy', verbose=1)

    model.learn(total_timesteps=steps, callback=reward_callback)

    return model
from wrapper import ImageOnlyWrapper
from dataset import generate_teacher_dataset_to_disk, TeacherDiskDataset
from models import SupervisedPolicy, ImageOnlyExtractor
from train_supervised import train_student
from train_rl import train_student_with_rl
from evaluation import evaluate_policy, plot_rl_rewards


import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from env.custom_hopper import CustomHopper
from mujoco_py import GlfwContext

GlfwContext(offscreen=True)

def main(generate_dataset):
    num_episodes = 1000
    num_epochs = 20  

    dataset_name = f"distillation/outputs/teacher_dataset_{num_episodes}eps"
    student_policy_name = f"distillation/outputs/student_policy_{num_episodes}eps_{num_epochs}epochs.pt"
    rl_model_name = f"distillation/outputs/student_rl_finetuned_{num_episodes}eps_{num_epochs}epochs"

    # 1. Teacher
    train_env_state = Monitor(CustomHopper(domain='source'))
    train_env_image = ImageOnlyWrapper(Monitor(CustomHopper(domain='source')))
    test_env_image = ImageOnlyWrapper(Monitor(CustomHopper(domain='target')))

    teacher_model = PPO.load("ppo/tuned_ppo", env=train_env_state)

    # 2. Dataset
    if generate_dataset:
        print(f"Generating dataset ({num_episodes} episodes)...")
        generate_teacher_dataset_to_disk(
            teacher_model=teacher_model,
            env_state=train_env_state,
            env_image=train_env_image,
            output_dir=dataset_name,
            num_episodes=num_episodes
        )
    else:
        print(f"[!] Skipping dataset generation. Using: {dataset_name}")

    # 3. Student
    dataset = TeacherDiskDataset(dataset_name)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    if os.path.exists(student_policy_name) :
        print(f"[✓] Loading existing student policy and extractor...")
        student_policy = torch.load(student_policy_name) 
        student_policy.eval()
    else:
        print("[✓] Starting supervised training...")
        extractor = ImageOnlyExtractor()
        student_policy = SupervisedPolicy(extractor)
        train_student(student_policy, dataloader, epochs=num_epochs)
        torch.save(student_policy, student_policy_name)
        print("Student model saved.")


    # 4. RL fine-tuning
    if os.path.exists(f"{rl_model_name}.zip"):
        print(f"[✓] RL fine-tuned model found at {rl_model_name}.zip — skipping RL training.")
        rl_model = PPO.load(rl_model_name, env=train_env_image)
    else:
        rl_model = train_student_with_rl(student_model=student_policy)
        rl_model.save(rl_model_name)
        print(f"[✓] RL fine-tuned model saved to {rl_model_name}")
        plot_rl_rewards(save_path="distillation/outputs/rl_distillation_plot.png")

    # 5. Evaluation

    print("\n[Evaluation] Supervised policy on SOURCE domain:")
    evaluate_policy(student_policy, train_env_image, is_torch_model=True, device="cuda")

    print("\n[Evaluation] Fine_tuned policy on SOURCE domain:")
    evaluate_policy(rl_model, train_env_image)

    print("\n[Evaluation] Supervised policy on TARGET domain:")
    evaluate_policy(student_policy, test_env_image, is_torch_model=True, device="cuda")

    print("\n[Evaluation] Fine_tuned policy on TARGET domain:")
    evaluate_policy(rl_model, test_env_image)

if __name__ == "__main__":
    main(generate_dataset=False)
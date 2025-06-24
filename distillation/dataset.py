import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob

def generate_teacher_dataset_to_disk(teacher_model, env_state, env_image, output_dir="dataset_eps", num_episodes=500):
    os.makedirs(output_dir, exist_ok=True)

    for ep in range(num_episodes):
        obs_state = env_state.reset()
        obs_image = env_image.reset()

        images = []
        actions = []

        done_state = False
        done_image = False

        while not (done_state or done_image):
            action, _ = teacher_model.predict(obs_state, deterministic=True)

            images.append(obs_image.astype(np.float16))
            actions.append(action.astype(np.float32))

            try:
                obs_state, _, done_state, _ = env_state.step(action)
            except RuntimeError:
                print("env_state requires reset")
                break

            try:
                obs_image, _, done_image, _ = env_image.step(action)
            except RuntimeError:
                print("env_image requires reset")
                break

        episode_path = os.path.join(output_dir, f"ep_{ep:04d}.npz")
        np.savez_compressed(episode_path,
                            images=np.stack(images),
                            actions=np.stack(actions))
        print(f"Saved episode {ep+1}/{num_episodes} to {episode_path}")
    
    print("All episodes saved.")

class TeacherDiskDataset(Dataset):
    def __init__(self, folder_path):
        self.file_paths = sorted(glob.glob(os.path.join(folder_path, "*.npz")))
        self.samples = []

        for path in self.file_paths:
            data = np.load(path)
            n = len(data['images'])
            self.samples.extend([(path, i) for i in range(n)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, i = self.samples[idx]
        data = np.load(file_path)
        image = torch.tensor(data['images'][i], dtype=torch.float32)
        action = torch.tensor(data['actions'][i], dtype=torch.float32)
        return image, action
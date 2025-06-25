# Project on Reinforcement Learning (Course project MLDL 2025 - POLITO)
### Teaching assistants: Andrea Protopapa and Davide Buoso

This work explores the sim‑to‑real transfer in robotic control using Reinforcement Learning techniques via Domain Randomization and Visual-Based inputs on the MuJoco Hopper environment.

The following methods are implemented:
- REINFORCE 
- Actor-Critic 
- Proximal Policy Optimization (PPO)
- Domain Randomization
- Visual-Based Reinforcement Learning (visual-only approach, multi-modal approach, distillation)

### Project structure
The project contains the following folders:
- `env`: it contains `custom_hommer.py` about the Hopper environment
- `outputs_reinforce`: it contains the rewards of the evaluation of the REINFORCE method and the policy `reinforce.mdl`
- `outputs_reinforce_baseline`: it contains the rewards of the evaluation of the REINFORCE with baseline method and the policy `reinforce_baseline.mdl`
- `outputs_actor_critic`: it contains the rewards of the evaluation of the actor_critic method and the policy `actor_critic.mdl`
- `ppo`: it contains the folder `outputs`, where there are the files: `bounds.txt`, containing the bounds; `tuning_results.txt`, containing the results of the grid search on the best configuration; `tuning_results_randomization.txt`, containing the results of the grid search over the probability distributions. It also contains the policies `tuned_ppo.zip` and `randomized_ppo.zip`.
- `images`: it contains the wrapper classes and preprocessing files for visual-only and multi-modal training and the policies `ppo_combined.zip` and `ppo_img_only.zip`.
- `distillation`: it contains the folder `outputs`, where there are the policies for distillation and the classes and preprocessing files for knowledge distillation.

### Instructions

To train the REINFORCE, REINFORCE with baseline and Actor-Critic methods, the file `train.py` is used. It parses arguments from terminal, the possible arguments are: `--n-episodes`, to specify the number of episodes for training; `--print-every`, to print intermediate results during training; `--agent`, to choose the desired model; `--device`, whether to use cpu or cuda.

To evaluate the REINFORCE, REINFORCE with baseline and Actor-Critic methods, the file `test.py` is used. It parses arguments from terminal, the possible arguments are: `--agent`, to choose the desired model; `--device`, wheter to use cpu or cuda; `--render`, whether to render or not the simulation; `--episodes`, the number oof episodes used for evaluation.

To tune or evaluate bounds of the agent with the PPO algorithm without randomization, the file `sb3.py` is used. It parses arguments from terminal, the possible arguments are: `--policy`, to specify the desired policy (by default is PPO); `--task`, whether to tune or evaluate bounds for the agent with the found configuration.

To tune or evaluate the agent with the PPO algorithm with randomization, the file `sb3_randomization.py` is used. It parses an argument from terminal, the possible argument is: `--tuning`, to choose whether tune the agent to find the best distribution or evaluate the performances with the found distribution.

To train or evaluate the agent in the visual-only approach, the file `train_only_images.py` is used. To decide whether to train or evaluate the agent there is a variable inside the main called 'task', whose value can be 'train' or 'evaluate'.

To train or evaluate the agent in the multi-modal approach, the file `train_images_states.py` is used. To decide whether to train or evaluate the agent there is a variable inside the main called 'task', whose value can be 'train' or 'evaluate'.

To train or evaluate the agent in the distillation approach, the file `main.py` inside the folder `distillation` is used. When this script is run it automatically trains and evaluate agents based on the knowledge distillation technique. To generate the dataset, the 'generate_dataset' flag must be set to True. For both the distillation and PPO training steps, the script checks if a model with the correct path already exists. If it does, the model is loaded automatically and training is skipped.

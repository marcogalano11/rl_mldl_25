"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

from env.custom_hopper import *
import agent_reinforce as RE
import agent_actor_critic as AC
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100_000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20_000, type=int, help='Print info every <> episodes')
    parser.add_argument('--agent', default="reinforce", type=str, help='agent type [reinforce, reinforce_baseline, actor_critic]')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	
    return parser.parse_args()

args = parse_args()

def save_rewards(rewards, method_name):
    folder = f"outputs_{method_name}"
    filepath = f"{folder}/rewards_{method_name}.txt"

    os.makedirs(folder, exist_ok=True)

    with open(filepath, "w") as f:
        for r in rewards:
            f.write(f"{r}\n")

    print(f"Saved rewards to {filepath}")

def main():

	env = gym.make('CustomHopper-source-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	if args.agent == "reinforce":

		policy = RE.Policy(observation_space_dim, action_space_dim)
		agent = RE.Agent(policy, device=args.device)

	elif args.agent == "reinforce_baseline":

		baseline = 20
		policy = RE.Policy(observation_space_dim, action_space_dim)
		agent = RE.Agent(policy, device=args.device, baseline=baseline)

	elif args.agent == "actor_critic":
	
		policy = AC.Policy(observation_space_dim, action_space_dim)
		agent = AC.Agent(policy, device=args.device)

	else: 

		raise ValueError(f"Unsupported or wrong agent type: {args.agent}")

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

	rewards = []

	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities, value = agent.get_action(state, False)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, value)

			train_reward += reward
		
		agent.update_policy()
		print("here")

		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return: ', train_reward)
		
		rewards.append(train_reward)


	torch.save(agent.policy.state_dict(), f"outputs_{args.agent}/{args.agent}.mdl")

	save_rewards(rewards, args.agent)

	

if __name__ == '__main__':
	main()
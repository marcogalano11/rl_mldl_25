"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
import agent_reinforce as RE
import agent_actor_critic as AC

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', default='reinforce', type=str, help='model name [reinforce, reinforce_baseline, actor_critic]')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=50, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()

def plot_rewards(rewards, method_name):
	episodes = list(range(1, len(rewards) + 1))
	plt.plot(episodes, rewards)
	plt.xlabel('Episodes')
	plt.ylabel('Rewards')
	plt.title(f"Rewards over {len(episodes)} episodes of {method_name}")
	plt.show()

def main():

	env = gym.make('CustomHopper-source-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	if args.agent == "reinforce":

		policy = RE.Policy(observation_space_dim, action_space_dim)
		policy.load_state_dict(torch.load(f"{args.agent}.mdl"), strict=True)
		agent = RE.Agent(policy, device=args.device)

	elif args.agent == "reinforce_baseline":

		baseline = 20
		policy = RE.Policy(observation_space_dim, action_space_dim)
		policy.load_state_dict(torch.load(f"{args.agent}.mdl"), strict=True)
		agent = RE.Agent(policy, device=args.device, baseline=baseline)

	elif args.agent == "actor_critic":
	
		policy = AC.Policy(observation_space_dim, action_space_dim)
		policy.load_state_dict(torch.load(f"{args.agent}.mdl"), strict=True)
		agent = AC.Agent(policy, device=args.device)

	else: 

		raise ValueError(f"Unsupported or wrong agent type: {args.agent}")
	
	rewards = []

	for episode in range(args.episodes):
		done = False
		test_reward = 0
		state = env.reset()

		while not done:

			action, _, _ = agent.get_action(state, evaluation=True)

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			if args.render:
				env.render()

			test_reward += reward
		
		rewards.append(test_reward)

		print(f"Episode: {episode + 1} | Return: {test_reward}")
	
	print(f"Average reward: {np.mean(rewards)}, std: {np.std(rewards)}")
	
	plot_rewards(rewards, args.agent)
	

if __name__ == '__main__':
	main()
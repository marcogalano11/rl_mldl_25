"""Train an RL agent on the OpenAI Gym Hopper environment using
	Actor-critic algorithm
"""
import argparse
import torch
import gym
from env.custom_hopper import *
from agent_actor_critic import Agent, Policy
import numpy as np
import matplotlib.pyplot as plt	


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()
def save_plot_rewards(reward_list, title="Training Performance", filename="rewards_plot.png"):
    episodes = np.arange(1, len(reward_list)+1)
    plt.plot(episodes, reward_list)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #
	reward_list = []
	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities, value = agent.get_action(state, False)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done, value)

			train_reward += reward
		
		agent.update_policy()
		reward_list.append(train_reward)

		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)


	torch.save(agent.policy.state_dict(), "actor_critic.mdl")
	save_plot_rewards(reward_list, title="Actor-Critic Training Performance", filename="actor_critic_rewards.png")

	

if __name__ == '__main__':
	main()
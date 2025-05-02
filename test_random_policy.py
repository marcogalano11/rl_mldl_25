"""Test a random policy on the OpenAI Gym Hopper environment.

    
    TASK 1: Play around with this code to get familiar with the
            Hopper environment.

            For example:
                - What is the state space in the Hopper environment? Is it discrete or continuous? => continuous
                - What is the action space in the Hopper environment? Is it discrete or continuous? => continuous
                - What is the mass value of each link of the Hopper environment, in the source and target variants respectively? => check dynamic parameters after switching env creation from source to target, only torso changes
                - what happens if you don't reset the environment even after the episode is over? => The environment won't go to the next timestep without resetting the starting space, causing: AssertionError: Cannot call env.step() before calling reset()
                - When exactly is the episode over? => when the hopper is unhealty (fails task) or when timestep is reached
                - What is an action here? => a vector representing movements of the link between body parts (3 links between 4 body parts)
"""
import pdb

import gym

from env.custom_hopper import *


def main():
	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('State space:', env.observation_space) # state-space
	print('Action space:', env.action_space) # action-space
	print('Dynamics parameters:', env.get_parameters()) # masses of each link of the Hopper

	n_episodes = 500
	render = False

	for episode in range(n_episodes):
		done = False
		state = env.reset()	# Reset environment to initial state

		while not done:  # Until the episode is over

			action = env.action_space.sample()	# Sample random action
		
			state, reward, done, info = env.step(action)	# Step the simulator to the next timestep

			if render:
				env.render()

"""Final considerations: THe body of the hopper is made of four parts: [torso, thigh, leg, foot]. Between these 
   4 parts there are 3 links (hip, knee, ankle), that are the ones coordinated by the actions (in fact action 
   space is (3,)).
   The environment can be switched between source and target. Source is the more stable environment, where we train our agent.
   Target is the test environment, the one that should be representing the real world. They may differ in weights."""

	

if __name__ == '__main__':
	main()
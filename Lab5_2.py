# Lab5 Frozen Lake in stochastic environment
# Apply previous Q-learning method which is suitable just for deterministic environment
# Success rate should be very poor

# Added two lines below to avoid error message in ubuntu command line interface.
import matplotlib
matplotlib.use('Agg')

import gym
import numpy as np
import matplotlib.pyplot as plt
import random as pr

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000

rList = []

for i in range(num_episodes):
	state = env.reset()
	rAll = 0 # total reward
	done = False

	while not done:
		action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
		new_state, reward, done, _ = env.step(action)
		Q[state, action] = reward + np.max(Q[new_state, :])
		rAll += reward
		state = new_state
	
	rList.append(rAll)


print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()	

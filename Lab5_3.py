# Lab5 Frozen Lake in stochastic environment
# Apply new Q-learning method which is suitable for stochastic environment
# Success rate should be improved like 50 ~ 60% 

# Added two lines below to avoid error message in ubuntu command line interface.
import matplotlib
matplotlib.use('Agg')

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])

learning_rate = .85 # defined learning_rate
dis = .99
num_episodes = 2000

rList = []

for i in range(num_episodes):
	state = env.reset()
	rAll = 0 # total reward
	done = False

	while not done:
		action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
		new_state, reward, done, _ = env.step(action)
		Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate*(reward + dis*np.max(Q[new_state, :]))
		rAll += reward
		state = new_state
	
	rList.append(rAll)


print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()	

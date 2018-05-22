# Lab4 Frozen Lake - using epsilon-greedy (instead of random noise)

# Added two lines below to avoid error message in ubuntu command line interface.
import matplotlib
matplotlib.use('Agg')

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
	id='FrozenLake-v3',
	entry_point='gym.envs.toy_text:FrozenLakeEnv',
	kwargs={'map_name': '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n, env.action_space.n])
dis = .99 # added discount factor 
num_episodes = 2000

rList = []

for i in range(num_episodes):
	state = env.reset()
	rAll = 0 # total reward
	done = False
	e = 1. / ((i // 100) + 1) # defined epsilon

	while not done:
		# used epsilon-greedy
		# used np.argmax() instead of rargmax()
		if np.random.rand(1) < e:
			action = env.action_space.sample() # exploration
		else:
			action = np.argmax(Q[state, :]) # exploitation
		new_state, reward, done, _ = env.step(action)
		Q[state, action] = reward + dis*np.max(Q[new_state, :]) # multiplied discount factor
		rAll += reward
		state = new_state
	
	rList.append(rAll)


print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()	

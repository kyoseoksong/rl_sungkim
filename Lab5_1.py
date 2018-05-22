# Lab5 Frozen Lake in stochastic environment
# manual experiment with keyboard input

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B' : DOWN,
    '\x1b[C' : RIGHT,
    '\x1b[D' : LEFT
}

import gym
import readchar

env = gym.make("FrozenLake-v0") # In FrozenLake-v0, is_slippery is True, by default.
env.reset()
env.render()

while True:
    ## key = inkey()
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break
        
    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)
    
    if done:
        print("Finished with reward", reward)
        break

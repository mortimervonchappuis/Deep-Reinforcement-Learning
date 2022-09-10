# TODO
# NETWORKS
#	Representations
#	Dynamics
#	Policy/Value
# MCTS
# Environment

import gym
import gym_chess

#env = gym.make('Chess-v0')
env = gym.vector.make('CarRacing-v2', 2)
Os = env.reset()
print(env)

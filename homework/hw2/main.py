if __name__ == '__main__':
	from gridworld import *
	from agent import *

	#env = GridWORLD()
	#env.seed(42)
	#Sarsa = TabularQ(env, policy=EpsilonGreedy, alpha=0.3, gamma=0.99, N=float('inf'))
	#Sarsa.train(episodes=100, render={'mode': 'human', 'speed': 0.001})
	
	env = Labyrinth()
	env.seed(42)
	Sarsa = TabularSarsa(env, policy=EpsilonGreedy, alpha=0.3, gamma=0.99, N=10)
	Sarsa.train(episodes=100, render={'mode': 'human', 'speed': 0.001})
	
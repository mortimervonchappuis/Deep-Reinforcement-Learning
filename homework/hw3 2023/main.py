from gridworld import *
from agent import *
from policy import *
from matplotlib import pyplot as plt



if __name__ == '__main__':
	from gridworld import *
	from agent import *

	def average(Gs, shift=0.2):
		Gs_avr = []
		G_avr = Gs[0]
		for G in Gs:
			G_avr = shift * G + (1 - shift) * G_avr
			Gs_avr.append(G_avr)
		return Gs_avr

	def plot(Gs, Ts):
		Gs = average(Gs)
		plt.title('Returns over Episodes')
		plt.plot(range(len(Gs)), Gs)
		plt.xlabel('Episodes')
		plt.ylabel('Smoothed Returns')
		plt.show()
		plt.title('Returns over Time')
		plt.plot(Ts, Gs)
		plt.xlabel('Time')
		plt.ylabel('Smoothed Returns')
		plt.show()

	#env = GridWORLD()
	#env.seed(42)
	#Sarsa = TabularQ(env, policy=EpsilonGreedy, alpha=0.3, gamma=0.99, N=float('inf'))
	#Sarsa.train(episodes=100, render={'mode': 'human', 'speed': 0.001})
	
	env = Labyrinth()
	env.seed(42)

	agent = TabularSarsa(env, policy=EpsilonGreedy, alpha=0.3, gamma=0.99, N=3)
	#agent = TabularMC(env, policy=Softmax, alpha=0.3, gamma=0.99)

	#Gs, Ts = agent.train(episodes=100, render={'mode': None, 'speed': 0})
	Gs, Ts = agent.train(episodes=100, render={'mode': ('human', 'q'), 'speed': 0})
	
	plot(Gs, Ts)


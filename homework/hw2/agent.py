from collections import defaultdict
from time import sleep
from random import randrange
from policy import *
import numpy as np
from pickle import dump, load



class TabularAgent:
	Q = defaultdict(lambda: 0.0)
	V = defaultdict(lambda: 0.0)


	def __call__(self, render={'mode':'human', 'speed': 0.1}):
		O, R = env.reset(), 0
		done = False
		while not done:
			self.render(render)
			A = self.policy.draw(O)
			O, R, done, info = self.env.step(A) # (X, Y)
		self.render(render)


	def save(self, path):
		with open(path, 'wb') as file:
			dump((dict(self.Q), dict(self.V)), file)


	def load(self, path):
		with open(path, 'rb') as file:
			Q, V = load(file)
			self.Q.update(Q)
			self.V.update(V)


	def train(self, episodes, render={'mode':'human', 'speed': 0}):
		for n in range(episodes):
			O, R = self.env.reset(), 0
			done = False
			self.pre_episode()
			while not done:
				self.render(render)
				print(f'EPISODE {n}')
				A = self.action(O, R)
				O, R, done, info = self.env.step(A)
			self.render(render)
			print(f'EPISODE {n}')
			self.post_episode(O, R)


	def render(self, render):
		speed = render['speed']
		mode = render['mode']
		if speed:
			sleep(speed)
		if mode:
			self.env.render(mode)


	def expectation(self, O):
		return sum(self.policy(O, A) * self.Q[O, A] for A in self.env.action_space)


	def action(self, O, R):
		return self.env.action_space.sample()


	def pre_episode(self,):
		pass


	def post_episode(self,):
		pass



class TabularTD(TabularAgent):
	def __init__(self, env, alpha=0.5, gamma=.9, policy=Softmax, N=10):
		from matplotlib import pyplot as plt
		self.alpha = alpha
		self.gamma = gamma
		self.env = env
		self.As = range(self.env.action_space.n)
		self.policy = policy(self)
		self.N = N
		self.plt = plt
		self.plt.ion()


	def Qviz(self):
		X = self.env.rows
		Y = self.env.columns
		Vs = np.zeros((X + 2, Y + 2))
		N = np.zeros((X + 2, Y + 2))
		for k in range(4):
			for i in range(X):
				for j in range(Y):
					if k == 0:
						Vs[i, j + 1] += self.Q[(i, j), k]
						if self.Q[(i, j), k] != 0: # NORTH
							N[i, j + 1] += 1
					elif k == 1:
						Vs[i + 1, j + 2] += self.Q[(i, j), k]
						if self.Q[(i, j), k] != 0: # EAST
							N[i + 1, j + 2] += 1
					elif k == 2:
						Vs[i + 2, j + 1] += self.Q[(i, j), k]
						if self.Q[(i, j), k] != 0: # SOUTH
							N[i + 2, j + 1] += 1
					elif k == 3:
						Vs[i + 1, j] += self.Q[(i, j), k]
						if self.Q[(i, j), k] != 0: # WEST
							N[i + 1, j] += 1
		self.plt.imshow(Vs[1:X+1, 1:Y+1]/N[1:X+1, 1:Y+1], cmap='inferno')
		self.plt.draw()
		self.plt.pause(0.001)


	def pre_episode(self):
		self.trajectory = []
		self.t = 0


	def action(self, O_tn, R_tn, done=False):
		A_tn = self.policy.draw(O_tn)
		self.t += 1
		if self.t > self.N or done:
			O_t, R_t, A_t = self.trajectory.pop(0)
			S = sum(self.gamma**k * R for k, (O, R, A) in enumerate(self.trajectory)) 
			G = S + self.gamma**len(self.trajectory) * self.Q[O_tn, A_tn]
			self.update(G, O_t, A_t)
		self.trajectory.append((O_tn, R_tn, A_tn))
		return A_tn


	def post_episode(self, O, R):
		N = len(self.trajectory)
		for n in range(N):
			A = self.action(O, R, done=True)
			O, R, done, info = self.env.step(A)
		self.Qviz()



class TabularSarsa(TabularTD):
	def update(self, G, O, A):
		self.Q[O, A] += self.alpha * (G - self.Q[O, A])



class TabularQ(TabularTD):
	def update(self, G, O, A):
		self.Q[O, A] += self.alpha * (G - max(self.Q[O, a] for a in self.As))



if __name__ == '__main__':
	from gridworld import Gridworld, Labyrinth

	env = Gridworld()
	#env = Labyrinth()
	agent = TabularSarsa(env, policy=EpsilonGreedy, alpha=0.3, gamma=0.99, N=5)
	#agent = TabularQ(env, policy=Softmax, alpha=0.3, gamma=0.98, N=5)
	agent.train(episodes=100, render={'mode': 'human', 'speed': 0.001})
	#agent.save('gridworld_q.pkl')
	#agent.load('gridworld_q.pkl')
	#agent(render={'mode': 'human', 'speed': 0.1})
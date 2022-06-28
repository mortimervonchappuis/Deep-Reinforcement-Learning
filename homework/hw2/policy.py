import random
import numpy as np



class Policy:
	def __init__(self, agent):
		self.agent = agent
		self.As = range(agent.env.action_space.n)
		self.Q = self.agent.Q
		self.V = self.agent.V


	def draw(self, O):
		P = self.distribution(O)
		return random.choices(self.As, P)[0]


	def __call__(self, O, A):
		P = self.distribution(O)
		return P[A]


class Softmax(Policy):
	def distribution(self, O):
		Qs = np.array([self.Q[O, A] for A in self.As])
		exp = np.exp(Qs)
		return exp/np.sum(exp)



class EpsilonGreedy(Policy):
	epsilon = 0.2


	def distribution(self, O):
		A = max(self.As, key=lambda a: self.Q[O, a])
		return np.array([self.epsilon/len(self.As) 
			if a != A else 1 - self.epsilon + self.epsilon/len(self.As)
			for a in self.As ])

		
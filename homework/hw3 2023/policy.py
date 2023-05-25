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
		A = random.choices(self.As, P)[0]
		if self.agent.fixed and self.agent.t == 0:
			O, A = self.agent.fixed
		return A


	def __call__(self, O, A):
		P = self.distribution(O)
		return P[A]


class Softmax(Policy):
	temperatur = 100
	max_prob = 0.9


	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.min_prob = (1 - self.max_prob) / (len(self.As) - 1)


	def distribution(self, O):
		Qs = np.array([self.Q[O, A] for A in self.As])
		exp = np.exp(Qs) ** (1/self.temperatur)
		exp = np.nan_to_num(exp, nan=1.0)#, posinf=1.0, neginf=0.0)
		P = exp/np.sum(exp)
		P = np.maximum(np.minimum(P, self.max_prob), self.min_prob)
		return P



class EpsilonGreedy(Policy):
	epsilon = 0.2


	def distribution(self, O):
		A = max(self.As, key=lambda a: self.Q[O, a])
		return np.array([self.epsilon/len(self.As) 
			if a != A else 1 - self.epsilon + self.epsilon/len(self.As)
			for a in self.As ])

		
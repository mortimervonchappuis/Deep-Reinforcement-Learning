import numpy as np
import gym
import random
import pickle


class ReplayBuffer:
	def __init__(self, size, env):
		self.o_space ,= env.observation_space.shape
		self.a_space  = env.action_space.n
		self.size     = size
		self.n        = 0
		self.Os_init  = np.empty((size, self.o_space), dtype=np.float64)
		self.As 	  = np.empty((size), 			   dtype=np.int32)
		self.Rs		  = np.empty((size), 			   dtype=np.float64)
		self.Os_next  = np.empty((size, self.o_space), dtype=np.float64)
		self.Ds 	  = np.empty((size))


	def __iadd__(self, samples):
		n = len(samples)
		Os_init, As, Rs, Os_next, Ds = zip(*samples)
		self.Os_init = np.roll(self.Os_init, n, axis=0)
		self.As		 = np.roll(self.As, 	 n, axis=0)
		self.Rs		 = np.roll(self.Rs, 	 n, axis=0)
		self.Os_next = np.roll(self.Os_next, n, axis=0)
		self.Ds 	 = np.roll(self.Ds, 	 n, axis=0)
		self.Os_init[:n,...] = np.array(Os_init)
		self.As[:n,...]		 = np.array(As)
		self.Rs[:n,...]		 = np.array(Rs)
		self.Os_next[:n,...] = np.array(Os_next)
		self.Ds[:n,...] 	 = np.array(Ds)
		self.n = min(self.size, self.n + n)
		return self


	def __getitem__(self, index):
		Os_init = self.Os_init[index,...]
		As		= self.As[index,...]
		Rs		= self.Rs[index,...]
		Os_next = self.Os_next[index,...]
		Ds 		= self.Ds[index,...]
		return Os_init, As, Rs, Os_next, Ds


	@property
	def full(self):
		return self.size == self.n


	def add(self, samples):
		self += samples


	def batch(self, batch_size):
		indicies = random.sample(range(self.n), batch_size)
		return self[indicies]


	def save(self, path):
		with open(path, 'wb') as file:
			pickle.dump([self.Os_init, self.As, self.Rs, self.Os_next, self.Ds], file)


	def load(self, path):
		with open(path, 'rb') as file:
			self.Os_init, self.As, self.Rs, self.Os_next, self.Ds = pickle.load(file)
			self.n = self.Rs.shape[0]



if __name__ == '__main__':
	env = gym.make("LunarLander-v2")
	b = ReplayBuffer(100, env)
	o_sample = env.observation_space.sample()
	a_sample = env.action_space.sample()
	print(env.observation_space)
	print(env.action_space)
	print(a_sample)
	r_sample = 0
	samples = [(o_sample, a_sample, r_sample, o_sample)] * 3
	b += samples
	print(b.Os_next.shape)
	print(env.action_space.shape)

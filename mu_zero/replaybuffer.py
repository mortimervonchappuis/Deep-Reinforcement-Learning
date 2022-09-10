import numpy as np
import random
from pickle import load, dump
from os.path import isfile



class ReplayBuffer:
	def __init__(self, 
				 size, 
				 manager, 
				 K,
				 reinit=True, 
				 file_name='RPDB.pkl'):
		self.size      = size
		self.manager   = manager
		self.K         = K
		self.file_name = file_name
		if reinit:
			print('REINITIALIZED REPLAY BUFFER')
			self.DB = self.manager.list()
		else:
			if isfile(file_name):
				with open(file_name, 'rb') as file_input:
					self.DB = self.manager.list(load(file_input))
				print('RELOADED REPLAY BUFFER')
			else:
				raise FileNotFoundError(f'{file_name} does not exsist')


	def __len__(self):
		return len(self.DB)


	def __getitem__(self, key):
		return self.DB[key]


	def __setitem__(self, key, value):
		self.DB[key] = value


	def extend(self, data):
		self.DB.extend(data)
		if len(self) > self.size:
			#del self.DB[0]
			self.DB.pop()


	def save(self):
		with open(self.file_name, 'wb') as file_output:
			data = list(self.DB)
			dump(data, file_output)


	def fetch(self, batch_size):
		Os, As, PIs, Us, Zs = [], [], [], [], []
		T = np.array([len(trajectory) - self.K for trajectory in self.DB])
		P = T/T.sum()
		for i in np.random.choice(len(self), batch_size, p=P):
			trajectory = self[i]
			j = random.choice(range(len(trajectory)) - self.K)
			O, A, PI, U, Z = zip(trajectory[j:j + self.K])
			Os.append(O[0])
			As.append(A)
			PIs.append(PI)
			Us.append(U)
			Zs.append(Z)
		return np.array(Os), np.array(As), np.array(PIs), np.array(Us), np.array(Zs)


	def __bool__(self):
		return len(self) >= self.size



if __name__ == '__main__':
	from multiprocessing import Manager
	m = Manager()
	RP = ReplayBuffer(size=10, manager=m)
	for i in range(20):
		RP.extend([(np.zeros((2,)), np.ones((1,)), i)])
	RP.save()

	RP = ReplayBuffer(size=10, manager=m, reinit=False)
	print(RP.fetch(batch_size=4))
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from math import tau, sqrt
import numpy as np
import gym



class Vanilla: # BORING
	def __init__(self, env_name, n_proc, actor, gamma=0.99, decay=0.9):
		self.envs   = gym.vector.make(env_name, n_proc)
		self.n_proc = n_proc
		self.actor  = actor(self.envs)
		self.gamma  = gamma
		self.decay  = decay
		self.buffer = [{'O': [], 'R': [], 'A': []} for _ in range(self.n_proc)]


	def __call__(self, episodes, max_t=float('inf')):
		O = self.envs.reset()
		self.scores = []
		score_max = float('-inf')
		with tqdm(total=episodes) as bar:
			episode = 0
			n = 0
			while episode < episodes:
				bar.set_description('SAMPLING')
				A, P = self.actor.actions(O)
				#As = np.nan_to_num(As)
				O_next, R, D, _ = self.envs.step(A)
				for i in range(self.n_proc):
					self.buffer[i]['O'].append(O[i])
					self.buffer[i]['A'].append(A[i])
					self.buffer[i]['R'].append(R[i])
				O = O_next
				n += 1
				if np.any(D) or n % max_t == 0:
					bar.set_description('TRAINING')
					if np.any(D):
						bar.update(1)
						episode += 1
					with tf.GradientTape() as tape:
						target = 0
						self.G_avr = sum(self.gamma**t * R for j in range(self.n_proc) \
										 for t, R in enumerate(self.buffer[j]['R']))/self.n_proc
						score = sum(sum(x['R']) for x in self.buffer)/self.n_proc
						if score > score_max:
							self.actor.save_weights('breakthrough.pd')
							score_max = score
						self.scores.append(score)
						for i in range(self.n_proc):
							# REINFORCE UPDATE
							G = sum(self.gamma**t * R for t, R in enumerate(self.buffer[i]['R']))
							Os = np.array(self.buffer[i]['O'])
							As = np.array(self.buffer[i]['A'])
							Ps = self.actor.log_prob(Os, As)
							Gs = np.empty(len(self.buffer[i]['R']))
							for t, R in enumerate(self.buffer[i]['R']):
								Gs[t] = (G - self.G_avr) * self.gamma**t
								G = (G - R)/self.gamma
							assert Gs.shape == Ps.shape
							target += tf.reduce_mean(-Gs * Ps)
							self.buffer[i] = {'O': [], 'R': [], 'A': []}
						gradients = tape.gradient(target/self.n_proc, self.actor.trainable_weights)
						self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_weights))
						bar.set_postfix({'G': self.G_avr, 'Score': score, 'J': np.max(target)})
					if np.any(D):
						O = self.envs.reset()
		self.envs.close()
		return self.scores


	def show(self, env):
		Os = env.reset()
		D = False
		while not D:
		  As, Ps = self.actor.actions(Os[None, ...])
		  As = np.nan_to_num(As[0,...])
		  Os, Rs, D, _ = env.step(As)
		  env.render()




if __name__ == '__main__':
	from model import Actor
	from matplotlib import pyplot as plt

	agent = Vanilla(env_name="CarRacing-v1", n_proc=16, actor=Actor)
	env = gym.make("CarRacing-v1")
	#agent.actor.load_weights('vanilla_768_nobase.pd')
	#agent.show(env)

	#agent.actor.load_weights('vanilla_1024_bases.pd')
	#history = agent(1024)
	#agent.actor.save_weights('vanilla_1024_bases.pd')
	#plt.plot(range(len(history)), history)
	#plt.show()

	agent.actor.load_weights('breakthrough.pd')
	agent.show(env)

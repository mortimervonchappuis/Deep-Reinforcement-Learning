import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from math import tau, sqrt



class Vanilla:
	def __init__(self, env, actor, gamma=0.999, decay=0.9, max_t=None):
		self.env = env
		self.n_envs = env.num_envs
		self.actor = actor
		self.gamma = gamma
		self.decay = decay
		self.max_t = max_t
		self.buffer = [{'O': [], 'R': [], 'A': []} for _ in range(self.n_envs)]


	def __call__(self, episodes):
		Os = self.env.reset()
		self.G_avr = None
		with tqdm(total=episodes) as bar:
			episode = 0
			while episode < episodes:
				As, Ps = self.actor.actions(Os)
				Os, Rs, Ds, _ = self.env.step(As)
				for i in range(self.n_envs):
					self.buffer[i]['O'].append(Os[i])
					self.buffer[i]['A'].append(As[i])
					self.buffer[i]['R'].append(Rs[i])
				for i in np.arange(self.env.num_envs)[Ds.astype(bool)]:
					bar.update(1)
					episode += 1
					# REINFORCE UPDATE
					G = sum(self.gamma ** t * R for t, R in enumerate(self.buffer[i]['R']))
					if self.G_avr is None:
						self.G_avr = G
					else:
						self.G_avr = self.G_avr * self.decay + (1 - self.decay) * G
					for t, (O, A, R) in enumerate(zip(self.buffer[i]['O'], 
													  self.buffer[i]['A'], 
													  self.buffer[i]['R'])):
						with tf.GradientTape() as tape:
							P = self.actor.log_prob(O[None, ...], A[None, ...])
							target = G * self.gamma**t * P
							gradients = tape.gradient(-target, self.actor.trainable_weights)
						print()
						print(gradients)
						quit()
						self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_weights))
						G = (G - R)/self.gamma
					self.buffer[i] = {'O': [], 'R': [], 'A': []}
				bar.set_postfix({'G': self.G_avr})
		self.env.close()




if __name__ == '__main__':
	import gym
	from model import Actor
	env = gym.vector.make("CarRacing-v1", num_envs=2)

	
	actor = Actor(env)
	agent = Vanilla(env, actor)
	agent(3)
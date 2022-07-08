import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from math import tau, sqrt
import numpy as np
import gym



class Vanilla: # BORING
	def __init__(self, env_name, n_proc, actor, gamma=0.999, decay=0.99, max_t=None):
		self.envs   = gym.vector.make(env_name, n_proc)
		self.n_proc = n_proc
		self.actor  = actor(self.envs)
		self.gamma  = gamma
		self.decay  = decay
		self.max_t  = max_t
		self.buffer = [{'O': [], 'R': [], 'A': []} for _ in range(self.n_proc)]


	def __call__(self, episodes):
		Os = self.envs.reset()
		self.G_avr = None
		with tqdm(total=episodes) as bar:
			episode = 0
			while episode < episodes:
				bar.set_description('SAMPLING')
				As, Ps = self.actor.actions(Os)
				#As = np.nan_to_num(As)
				Os, Rs, Ds, _ = self.envs.step(As)
				for i in range(self.n_proc):
					self.buffer[i]['O'].append(Os[i])
					self.buffer[i]['A'].append(As[i])
					self.buffer[i]['R'].append(Rs[i])
				if np.any(Ds):
					bar.set_description('TRAINING')
					bar.update(1)
					episode += 1
					with tf.GradientTape() as tape:
						target = 0
						for i in np.arange(self.n_proc):
							# REINFORCE UPDATE
							G = sum(self.gamma**t * R for t, R in enumerate(self.buffer[i]['R']))
							if self.G_avr == None: self.G_avr = G
							self.G_avr = self.G_avr * self.decay + (1 - self.decay) * G
							Os = np.array(self.buffer[i]['O'])
							As = np.array(self.buffer[i]['A'])
							Ps = self.actor.log_prob(Os, As)
							Gs = np.empty(len(self.buffer[i]['R']))
							for t, R in enumerate(self.buffer[i]['R']):
								Gs[t] = G * self.gamma**t
								G = (G - R)/self.gamma
							target += tf.reduce_mean(-Gs * Ps)
							self.buffer[i] = {'O': [], 'R': [], 'A': []}
						gradients = tape.gradient(target/self.n_proc, self.actor.trainable_weights)
						self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_weights))
						bar.set_postfix({'G': self.G_avr, 'A': np.max(As), 'J': np.max(target)})
					Os = self.envs.reset()
		self.envs.close()


	def show(self, env):
		Os = env.reset()
		D = False
		while not D:
		  As, Ps = self.actor.actions(Os[None, ...])
		  print(As)
		  As = np.nan_to_num(As[0,...])
		  Os, Rs, D, _ = env.step(As)
		  env.render()




if __name__ == '__main__':
	from model import Actor

	agent = Vanilla(env_name="CarRacing-v1", n_proc=4, actor=Actor)
	env = gym.make("CarRacing-v1")
	agent.actor.load_weights('vanilla_init.pd')
	#agent.show(env)
	agent(16)
	#agent.actor.save_weights('vanilla_16_base.pd')
	agent.actor.load_weights('vanilla_16_base.pd')
	agent.show(env)

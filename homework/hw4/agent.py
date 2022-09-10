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
							self.actor.save_weights('PG_best.pd')
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



class PPO:
	def __init__(self, env_name, n_proc, actor, critic, 
		gamma=0.99, epsilon=0.2, target_KL=0.01):
		self.envs      = gym.vector.make(env_name, n_proc)
		self.n_proc    = n_proc
		self.actor     = actor(self.envs)
		self.critic    = critic(self.envs)
		self.gamma     = gamma
		self.epsilon   = epsilon
		self.target_KL = target_KL
		self.buffer    = [{'O_one': [], 'R': [], 'A': [], 'O_two': []} for _ in range(self.n_proc)]


	def __call__(self, episodes, iterations=80):
		self.scores = []
		score_max = float('-inf')
		with tqdm(total=episodes) as bar:
			episode = 0
			for episode in range(episodes):
				# GENERATE EPISODE
				done = False
				bar.set_description('SAMPLING')
				Os = self.envs.reset()
				while not done:
					As, Ps = self.actor.actions(Os)
					Os_next, Rs, Ds, _ = self.envs.step(As)
					for i in range(self.n_proc):
						self.buffer[i]['O_one'].append(Os[i])
						self.buffer[i]['A'].append(As[i])
						self.buffer[i]['R'].append(Rs[i])
						self.buffer[i]['O_two'].append(Os_next[i])
					Os = Os_next
					done = np.any(Ds)
				score = sum(sum(x['R']) for x in self.buffer)/self.n_proc
				if score > score_max:
					self.actor.save_weights('PPO_best_actor.pd')
					self.critic.save_weights('PPO_best_critic.pd')
					score_max = score
				self.scores.append(score)
				# UPDATES
				bar.set_description('TRAINING POLICY')
				policy_loss = self.actor_update(iterations)
				bar.set_description('TRAINING VALUE')
				value_loss  = self.critic_update(iterations)
				# MARK STATS
				bar.update(1)
				bar.set_postfix({'Score': score, 'J': policy_loss, 'L': value_loss})
				for i in range(self.n_proc):
					self.buffer[i] = {'O_one': [], 'R': [], 'A': [], 'O_two': []}
		self.envs.close()
		return self.scores


	def actor_update(self, iterations):
		Os_one_k, As_k, Rs_k, Os_two_k = [], [], [], []
		for i in range(self.n_proc):
			Os_one_k.extend(self.buffer[i]['O_one'])
			As_k.extend(self.buffer[i]['A'])
			Rs_k.extend(self.buffer[i]['R'])
			Os_two_k.extend(self.buffer[i]['O_two'])
		Os_one_k = np.array(Os_one_k)
		As_k	 = np.array(As_k)
		Rs_k	 = np.array(Rs_k)
		Os_two_k = np.array(Os_two_k)
		Vs_one	 = self.critic(Os_one_k)
		Vs_two	 = self.critic(Os_two_k)
		adv 	 = Rs_k + self.gamma * Vs_two - Vs_one
		Ps_k 	 = self.actor.prob(Os_one_k, As_k)
		Os_k 	 = Os_one_k
		for n in range(iterations):
			with tf.GradientTape() as tape:
				# CHECK KL DIVERGENCE
				Ps = self.actor.prob(Os_k, As_k)
				KL = tf.reduce_sum(Ps_k * tf.math.log(Ps_k) - Ps_k * tf.math.log(Ps))
				if KL > self.target_KL:
					break
				# POLICY UPDATE
				ratio = Ps / Ps_k
				clip = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
				target = tf.math.minimum(adv * ratio, adv * clip)
				target = tf.reduce_mean(-target)
				gradients = tape.gradient(target, self.actor.trainable_weights)
				self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_weights))
		return target.numpy()


	def critic_update(self, iterations):
		T  = len(self.buffer[0]['R'])
		Gs = np.empty((self.n_proc * T,))
		Os = np.empty((self.n_proc * T,) + self.buffer[0]['O_one'][0].shape)
		for i in range(self.n_proc):
			Os[i * T:(i + 1) * T,...] = self.buffer[i]['O_one']
			G = sum(self.gamma**t * R for t, R in enumerate(self.buffer[i]['R']))
			for t, R in enumerate(self.buffer[i]['R']):
				Gs[i * T + t] = G
				G = (G - R)/self.gamma
		for n in range(iterations):
			with tf.GradientTape() as tape:
				Vs = self.critic(Os)
				loss = tf.math.squared_difference(Vs, Gs)
				loss = tf.reduce_mean(loss)
				gradients = tape.gradient(loss, self.critic.trainable_weights)
				self.critic.optimizer.apply_gradients(zip(gradients, self.critic.trainable_weights))
		return loss.numpy()


	def show(self, env):
		Os = env.reset()
		D = False
		while not D:
		  As, Ps = self.actor.actions(Os[None, ...])
		  As = np.nan_to_num(As[0,...])
		  Os, Rs, D, _ = env.step(As)
		  env.render()





if __name__ == '__main__':
	from model import Actor, Critic
	from matplotlib import pyplot as plt

	env = gym.make("CarRacing-v2")

	# VANILLA POLICY GRADIENTS

	#agent = Vanilla(env_name="CarRacing-v1", n_proc=16, actor=Actor)
	#history = agent(256)
	#agent.actor.load_weights('PG_best.pd')
	#plt.plot(range(len(history)), history)
	#plt.show()

	# PROXIMAL POLICY OPTIMIZATION

	agent = PPO(env_name="CarRacing-v2", n_proc=16, actor=Actor, critic=Critic)
	agent.actor.load_weights('PPO_best_actor.pd')
	agent.critic.load_weights('PPO_best_critic.pd')
	#history = agent(128, 20)
	#plt.plot(range(len(history)), history)
	#plt.show()

	agent.show(env)

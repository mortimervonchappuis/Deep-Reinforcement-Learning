import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import DQN
from replaybuffer import ReplayBuffer
from tqdm import tqdm
from gym.wrappers.record_video import RecordVideo
import random
import tensorflow as tf
import numpy as np



class AgentDQN:
	BUFFER_PATH = 'dqn_lunar_lander.erb'
	def __init__(self, 
				 env, 
				 gamma=0.99, 
				 tau=1e-3,
				 epsilon=1.0,
				 epsilon_decay=0.995,
				 epsilon_min=0.01,
				 buffer_size=100_000, 
				 avr_decay=0.95,
				 learning_rate=0.001, 
				 batch_size=64):
		env.seed(42)
		random.seed(42)
		tf.keras.utils.set_random_seed(42)
		# INITs
		self.env           = env
		self.Q_target      = DQN(env, learning_rate)
		self.Q             = DQN(env, learning_rate)
		self.Q_target.set_weights(self.Q.get_weights())
		self.action_space  = env.action_space.n
		self.buffer        = ReplayBuffer(buffer_size, env)
		self.gamma         = gamma
		self.epsilon       = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min   = epsilon_min
		self.tau           = tau
		self.avr_decay     = avr_decay
		self.learning_rate = learning_rate
		self.batch_size    = batch_size


	def __call__(self, O, sample=True):
		Qs = self.Q.predict(O[None,...], verbose=0)
		self.Qs = Qs
		if sample and random.random() < self.epsilon:
			A = random.randrange(self.action_space)
		else:
			A = tf.argmax(Qs, axis=1)
		return int(A)


	def show(self, plt=None):
		if plt is not None:
			from IPython import display
		O = self.env.reset()
		done = False
		G, t = 0, 0
		while not done:
			A = self(O, sample=False)
			O, R, done, _ = self.env.step(A)
			if plt is not None:
				plt.imshow(self.env.render(mode='rgb_array'))
				display.display(plt.gcf())    
				display.clear_output(wait=True)
			else:
				self.env.render()
			G += self.gamma**t * R
			t += 1
		return G


	def step(self, O_init, history=False, bar=None, max_t=1000):
		O_init = np.nan_to_num(O_init)
		A = self(O_init)
		O_next, R, done, _ = self.env.step(A)
		self.buffer += [(O_init, A, R, O_next, int(done))]
		O_init = O_next
		self.G += self.gamma**self.t * R
		self.score += R
		self.t += 1
		if done or self.t == max_t:
			if self.init:
				self.G_avr = self.G
				self.score_avr = self.score
				self.init = False
			else:
				self.G_avr = self.avr_decay * self.G_avr + (1 - self.avr_decay) * self.G
				self.score_avr = self.avr_decay * self.score_avr + (1 - self.avr_decay) * self.score
			self.G, self.t, self.score = 0, 0, 0
			if bar is not None:
				bar.set_postfix({'G': self.G_avr, 'S': self.score_avr, 'l': self.loss, 'epsilon': self.epsilon})
			if history:
				self.history['G'].append(self.G_avr)
				self.history['L'].append(self.loss)
				self.history['S'].append(self.score_avr)
			O_init = self.env.reset()
		return O_init, done



	def train(self, episodes, K, video_path=None):
		O = self.env.reset()
		self.G, self.t, self.score = 0, 0, 0
		self.loss = 0
		self.G_avr = 0
		self.score_avr = 0
		self.history = {'L': [], 'G': [], 'S': []}
		self.init = False
		if os.path.exists(self.BUFFER_PATH):
			self.buffer.load(self.BUFFER_PATH)
			self.G_init = True
		else:
			with tqdm(total=self.buffer.size) as bar:#
				bar.set_description('Filling Buffer')
				while not self.buffer.full:
					O, done = self.step(O)
					bar.update(1)
			self.buffer.save(self.BUFFER_PATH)
		with tqdm(total=episodes) as bar:
			bar.set_description('Training Agent')
			for episode in range(episodes):
				done = False
				self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
				while not done:
					for k in range(K):
						O, done = self.step(O, history=True, bar=bar)
					if self.buffer.n < self.batch_size:
						continue
					Os_init, As, Rs, Os_next, Ds = self.buffer.batch(self.batch_size)
					Qs_target = self.Q_target.predict(Os_next, verbose=0)
					Q_max = tf.math.reduce_max(Qs_target, axis=1)
					#print('Q_max:', Q_max.numpy())
					y = tf.stop_gradient(Rs + (1 - Ds) * self.gamma * Q_max)
					with tf.GradientTape() as tape:
						Qs = tf.gather(self.Q(Os_init), As, axis=1, batch_dims=1)
						loss = tf.math.squared_difference(y, Qs)
					gradients = tape.gradient(loss, self.Q.trainable_weights)
					self.loss = self.loss * self.avr_decay + (1 - self.avr_decay) * loss.numpy().mean()
					# POLYAK AVERAGING
					self.Q.optimizer.apply_gradients(zip(gradients, self.Q.trainable_weights))
					Q_target_weigths = self.Q_target.get_weights()
					Q_weights 		 = self.Q.get_weights()
					polyak_average   = [(1 - self.tau) * Q_t + self.tau * Q for Q_t, Q in zip(Q_target_weigths, Q_weights)]
					self.Q_target.set_weights(polyak_average)
				bar.update(1)
				if video_path is not None and episode % 100 == 0:
					try:
						self.construct_video(video_path, episode)
					except:
						print('FAIL VIDEO', episode)
		return self.history


	def construct_video(self, video_path, episode):
		env = RecordVideo(self.env, video_folder=video_path + f"/lunar_lander_{episode}")
		env.start_video_recorder()
		O = env.reset()
		done = False
		while not done:
			A = self(O)
			O, R, done, _ = env.step(A)
		env.close_video_recorder()





if __name__ == '__main__':
	import gym
	from matplotlib import pyplot as plt

	env = gym.make('LunarLander-v2')
	agent = AgentDQN(env, 
		buffer_size=100_000, 
		batch_size=64, 
		tau=1e-3, 
		gamma=0.99, 
		learning_rate=1e-3, 
		epsilon_decay=0.995, 
		epsilon_min=0.01, 
		epsilon=1)
	O = env.reset()
	os.system('clear')
	#print(agent.Q.predict(O[None,...]))
	#print(agent.Q_target.predict(O[None,...]))
	#quit()

	#agent.Q_target.save_weights('lunar_lander_init_e2000_target.pd')
	#agent.Q.save_weights('lunar_lander_init_e2000.pd')
	#agent.buffer.load('dqn_lunar_lander.erb')
	#agent.Q_target.load_weights('lunar_lander_e2000_target.pd')
	#agent.Q.load_weights('lunar_lander_e2000.pd')
	history = agent.train(episodes=2000, K=4, video_path='video') # inint K/10 N/10 G/-100
	agent.Q_target.save_weights('lunar_lander_e2000_target.pd')
	agent.Q.save_weights('lunar_lander_e2000.pd')
	agent.buffer.save('dqn_lunar_lander.erb')
	#agent.Q.load_weights('lunar_lander_e100_n100_k1.pd')
	#plt.plot(range(len(history)), history)
	#print('Return:', agent.show())
	#plt.show()
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from math import tau, sqrt, log



class Actor(Model):
	def __init__(self, env, learning_rate=1e-4, **kwargs):
		super().__init__(**kwargs)
		input_shape    = env.observation_space.shape[1:]
		output_shape   = env.action_space.shape[1:]
		self.optimizer = Adam(learning_rate=learning_rate)
		self.norm_one  = BatchNormalization()
		self.conv_one  = Conv2D(filters=16, kernel_size=[8, 8], strides=[4, 4],
								padding='same', activation='relu', name='conv_one')
		self.conv_two  = Conv2D(filters=32, kernel_size=[4, 4], strides=[2, 2],
								padding='same', activation='relu', name='conv_two')
		self.norm_two  = BatchNormalization()
		self.flatten   = Flatten()
		self.linear    = Dense(256, activation='relu', name='linear')
		self.mu        = Dense(*output_shape, activation='tanh', name='output_mu')
		self.sigma     = Dense(*output_shape, activation='elu', name='output_sigma')
		self.build((1, *input_shape))



	def call(self, Os):
		Xs = Os / 0xFF
		Xs = self.norm_one(Xs)
		Xs = self.conv_one(Xs)
		Xs = self.conv_two(Xs)
		Xs = self.flatten(Xs)
		Xs = self.linear(Xs)
		Mu = self.mu(Xs)
		Sigma = self.sigma(Xs)
		return Mu, Sigma + 1


	def log_prob(self, Os, As):
		Mu, Sigma = self(Os.astype(float))
		Ps_log = -tf.math.log(sqrt(tau) * Sigma) -1/2 * ((As - Mu)/Sigma)**2
		return tf.reduce_sum(Ps_log, axis=1)


	def prob(self, Os, As):
		return tf.math.exp(self.log_prob(Os, As))


	def actions(self, Os):
		Mu, Sigma = self(Os.astype(float))
		epsilon = tf.random.normal(Mu.shape)
		As = Mu + epsilon * Sigma
		Ps_log = -tf.math.log(sqrt(tau) * Sigma) -1/2 * ((As - Mu)/Sigma)**2
		return As.numpy(), tf.reduce_sum(Ps_log, axis=1).numpy()



class Critic(Model):
	def __init__(self, env, learning_rate=1e-3, **kwargs):
		super().__init__(**kwargs)
		input_shape    = env.observation_space.shape[1:]
		output_shape   = env.action_space.shape[1:]
		self.optimizer = Adam(learning_rate=learning_rate)
		self.norm_one  = BatchNormalization()
		self.conv_one  = Conv2D(filters=16, kernel_size=[8, 8], strides=[4, 4],
							   padding='same', activation='relu', name='conv_one')
		self.conv_two  = Conv2D(filters=32, kernel_size=[4, 4], strides=[2, 2],
							   padding='same', activation='relu', name='conv_two')
		self.norm_two  = BatchNormalization()
		self.flatten   = Flatten()
		self.linear    = Dense(256, activation='relu', name='linear')
		self.value     = Dense(1, activation=None, name='value')
		self.build((1, *input_shape))



	def call(self, Os):
		Xs = Os / 0xFF
		Xs = self.norm_one(Xs)
		Xs = self.conv_one(Xs)
		Xs = self.conv_two(Xs)
		Xs = self.flatten(Xs)
		Xs = self.linear(Xs)
		return self.value(Xs)[:,0]


if __name__ == '__main__':
	import gym
	env = gym.vector.make("CarRacing-v1", num_envs=2)

	print(env.action_space.shape)
	net = Actor(env)
	Os = env.observation_space.sample()#[None, ...]
	#print(Os)
	#print(net.predict(Os))
	print(net.actions(Os))
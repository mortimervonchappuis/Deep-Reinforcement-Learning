from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions as tfd
import tensorflow as tf
from math import tau, sqrt



class Actor(Model):
	def __init__(self, env, learning_rate=1e-3, **kwargs):
		super().__init__(**kwargs)
		input_shape    = env.observation_space.shape[1:]
		output_shape   = env.action_space.shape[1:]
		self.norm_one  = BatchNormalization()
		self.conv_one  = Conv2D(filters=16, kernel_size=[8, 8], strides=[4, 4],
							   padding='same', activation='relu', name='conv_one')
		self.conv_two  = Conv2D(filters=32, kernel_size=[4, 4], strides=[2, 2],
							   padding='same', activation='relu', name='conv_two')
		self.norm_two  = BatchNormalization()
		self.flatten   = Flatten()
		self.linear    = Dense(256, activation='relu', name='linear')
		self.mu        = Dense(*output_shape, activation=None, name='output_mu')
		self.sigma     = Dense(*output_shape, activation=None, name='output_sigma')
		self.optimizer = Adam(learning_rate=learning_rate)
		self.build((1, *input_shape))



	def call(self, Os):
		Xs = self.norm_one(Os)
		Xs = self.conv_one(Xs)
		Xs = self.conv_two(Xs)
		Xs = self.norm_two(Xs)
		Xs = self.flatten(Xs)
		Xs = self.linear(Xs)
		Mu = self.mu(Xs)
		Sigma = self.sigma(Xs)
		return Mu, Sigma


	def log_prob(self, Os, As):
		Mu, Sigma = self(Os)
		Ps = 1/(sqrt(tau) * Sigma) * tf.exp(-1/2 * ((As - Mu)/Sigma)**2)
		return tf.reduce_sum(tf.math.log(Ps))
		#N = tfd.MultivariateNormalDiag(loc=Mu, scale_diag=Sigma)
		#return N.log_prob(As)


	def actions(self, Os):
		Mu, Sigma = self.call(Os)
		N  = tfd.MultivariateNormalDiag(loc=Mu, scale_diag=Sigma)
		As = N.sample()
		Ps = N.log_prob(As)
		return As.numpy(), Ps.numpy()



if __name__ == '__main__':
	import gym
	env = gym.vector.make("CarRacing-v1", num_envs=2)

	print(env.action_space.shape)
	net = Actor(env)
	Os = env.observation_space.sample()#[None, ...]
	#print(Os)
	#print(net.predict(Os))
	print(net.actions(Os))
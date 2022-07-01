from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, ReLU
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam




class DQN(Model):
	def __init__(self, env, learning_rate, **kwargs):
		super().__init__(**kwargs)
		input_shape            = env.observation_space.shape
		output_shape           = env.action_space.n
		self.linear_hidden_one = Dense(150, activation=LeakyReLU(alpha=0.2))
		self.linear_hidden_two = Dense(120, activation=LeakyReLU(alpha=0.2))
		self.linear_out        = Dense(output_shape, activation=None)
		self.optimizer 	       = Adam(learning_rate=learning_rate)
		self.mse 			   = MeanSquaredError()
		self.build((1, *input_shape))


	def call(self, Os):
		Xs = self.linear_hidden_one(Os)
		Xs = self.linear_hidden_two(Xs)
		Qs = self.linear_out(Xs)
		return Qs



if __name__ == '__main__':
	import gym

	env = gym.make('LunarLander-v2')
	model = DQN(env)
	model.summary()
	print(model.predict(env.observation_space.sample()[None,...]))
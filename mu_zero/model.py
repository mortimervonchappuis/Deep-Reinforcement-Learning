import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import SeparableConv2D, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Layer
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np

Conv2D = SeparableConv2D

class ResidualBlock(Layer):
	def __init__(self, n_kernels, CONFIG,**kwargs):
		super().__init__(**kwargs)
		self.activation = ReLU()
		self.conv_1 = Conv2D(n_kernels, (3, 3), **CONFIG['conv'])
		self.batch_norm_1 = BatchNormalization()
		self.conv_2 = Conv2D(n_kernels, (3, 3), **CONFIG['conv'])
		self.batch_norm_2 = BatchNormalization()


	def call(self, inputs):
		x = self.conv_1(inputs)
		x = self.batch_norm_1(x)
		x = self.activation(x)
		x = self.conv_2(x)
		x = self.batch_norm_2(x)
		x = self.activation(x)
		x = self.activation(x + inputs)
		return x


	def get_config(self):
		base_config = super().get_config()
		return {**base_config}



class PolicyHead(Layer):
	def __init__(self, n_kernels, action_space, CONFIG, **kwargs):
		super().__init__(**kwargs)
		self.n_kernels = n_kernels
		self.CONFIG = CONFIG
		self.activation = ReLU()
		self.conv = Conv2D(n_kernels, (1, 1), **CONFIG['conv'])
		self.batch_norm = BatchNormalization()
		self.flatten = Flatten()
		self.dense = Dense(action_space, activation='softmax', **self.CONFIG['dense'])


	def call(self, inputs):
		p = self.conv(inputs)
		p = self.batch_norm(p)
		p = self.activation(p)
		p = self.flatten(p)
		p = self.dense(p)
		return p


	def get_config(self):
		base_config = super().get_config()
		return {**base_config, 
				'n_kernels': self.n_kernels, 
				'CONFIIG': self.CONFIG}


	def _preprocess(self, inputs):
		v = self.conv(inputs)
		v = self.batch_norm(v)
		return self.activation(v)



class StateHead(Layer):
	def __init__(self, n_kernels, CONFIG, output_shape=None, **kwargs):
		super().__init__(**kwargs)
		self.n_kernels = n_kernels
		self.CONFIG = CONFIG
		self.activation = ReLU()
		self.conv = Conv2D(n_kernels, (1, 1), **CONFIG['conv'])
		self.batch_norm = BatchNormalization()


	def build(self, input_shape):
		super().build(input_shape)


	def call(self, inputs):
		s = self.conv(inputs)
		s = self.batch_norm(s)
		s = self.activation(s)
		return s


	def get_config(self):
		base_config = super().get_config()
		return {**base_config, 
				'n_kernels': self.n_kernels, 
				'CONFIIG': self.CONFIG}



class ValueHead(Layer):
	def __init__(self, n_kernels, output_dim, CONFIG, **kwargs):
		print(kwargs)
		super().__init__(**kwargs)
		self.activation = ReLU()
		self.conv = Conv2D(n_kernels, (1, 1), **CONFIG['conv'])
		self.batch_norm = BatchNormalization()
		self.flatten = Flatten()
		self.dense_1 = Dense(output_dim, activation='relu', **CONFIG['dense'])
		self.dense_2 = Dense(1, activation=CONFIG['activation'], **CONFIG['dense'])


	def call(self, inputs):
		v = self.conv(inputs)
		v = self.batch_norm(v)
		v = self.activation(v)
		v = self.flatten(v)
		v = self.dense_1(v)
		v = self.dense_2(v)
		return v


	def get_config(self):
		base_config = super().get_config()
		return {**base_config}


	def _preprocess(self, inputs):
		v = self.conv(inputs)
		v = self.batch_norm(v)
		return self.activation(v)



class Representation(Model):
	CONFIG = {'conv': {'kernel_regularizer': l2(1e-4), 'padding': 'same'}, 
			  'dense': {'kernel_regularizer': l2(1e-4)}}
	def __init__(self, 
				 n_kernels=256,  
				 n_planes=16,
				 n_residual_blocks=16,
				 **kwargs):
		super().__init__(**kwargs)
		self.n_kernels = n_kernels
		self.conv = Conv2D(n_kernels, (3, 3), **self.CONFIG['conv'])
		self.batch_norm = BatchNormalization()
		self.activation = ReLU()
		self.residual_blocks = [ResidualBlock(n_kernels, self.CONFIG) \
								for n in range(n_residual_blocks - 1)]
		self.state_head = StateHead(n_planes, self.CONFIG)

	def call(self, inputs):
		x = self.conv(inputs)
		x = self.batch_norm(x)
		x = self.activation(x)
		for residual_block in self.residual_blocks:
			x = residual_block(x)
		x = self.state_head(x)
		return x


	def get_config(self):
		base_config = super().get_config()
		return {**base_config}


	def build(self, input_shape):
		self.state_head.build(input_shape)
		super().build(input_shape)



class Prediction(Model):
	CONFIG = {'conv': {'kernel_regularizer': l2(1e-4), 'padding': 'same'}, 
			  'dense': {'kernel_regularizer': l2(1e-4)}, 'activation': 'tanh'}
	def __init__(self, 
				 n_kernels=256, 
				 n_residual_blocks=20, 
				 n_value_units=128, 
				 action_space=4352,
				 **kwargs):
		super().__init__(**kwargs)
		self.n_kernels = n_kernels
		self.n_value_units = n_value_units
		self.conv = Conv2D(n_kernels, (3, 3), **self.CONFIG['conv'])
		self.batch_norm = BatchNormalization()
		self.activation = ReLU()
		self.residual_blocks = [ResidualBlock(n_kernels, self.CONFIG) \
								for n in range(n_residual_blocks - 1)]
		self.policy_head = PolicyHead(n_kernels, action_space, self.CONFIG)
		self.value_head = ValueHead(n_kernels, n_value_units, self.CONFIG)


	def call(self, inputs):
		x = self.tower(inputs)
		p = self.policy_head(x)
		v = self.value_head(x)
		return [p, v]


	def tower(self, inputs):
		x = self.conv(inputs)
		x = self.batch_norm(x)
		x = self.activation(x)
		for residual_block in self.residual_blocks:
			x = residual_block(x)
		return x


	def get_config(self):
		base_config = super().get_config()
		return {**base_config}


	def build(self, input_shape):
		self.policy_head.build(input_shape)
		self.value_head.build(input_shape)
		super().build(input_shape)


class Dynamics(Model):
	CONFIG = {'conv': {'kernel_regularizer': l2(1e-4), 'padding': 'same'}, 
			  'dense': {'kernel_regularizer': l2(1e-4)}, 'activation': 'tanh'}
	def __init__(self, 
				 n_kernels=256, 
				 n_residual_blocks=16, 
				 n_planes=16,
				 n_value_units=128, 
				 **kwargs):
		super().__init__(**kwargs)
		self.n_kernels = n_kernels
		self.n_value_units = n_value_units
		self.conv = Conv2D(n_kernels, (3, 3), **self.CONFIG['conv'])
		self.batch_norm = BatchNormalization()
		self.activation = ReLU()
		self.residual_blocks = [ResidualBlock(n_kernels, self.CONFIG) \
								for n in range(n_residual_blocks - 1)]
		self.state_head = StateHead(n_planes, self.CONFIG)
		self.value_head = ValueHead(n_kernels, n_value_units, self.CONFIG)
		


	def call(self, inputs):
		#print(*(i.shape for i in inputs))
		#inputs = tf.concat(inputs, axis=3)
		#print(inputs.shape)
		x = self.tower(inputs)
		s = self.state_head(x)
		v = self.value_head(x)
		return [s, v]


	def tower(self, inputs):
		x = self.conv(inputs)
		x = self.batch_norm(x)
		x = self.activation(x)
		for residual_block in self.residual_blocks:
			x = residual_block(x)
		return x


	def get_config(self):
		base_config = super().get_config()
		return {**base_config}


	def _build(self, input_shape):
		print('HI')
		print(input_shape)
		A, B = input_shape
		_input_shape = tf.TensorShape([a if i != 3 else a + b for i, (a, b) in enumerate(zip(A, B))])
		self.conv.build(_input_shape)

		self.state_head.build(_input_shape)
		self.value_head.build(_input_shape)
		super().build(input_shape)



class MuZeroNet(Model):
	def __init__(self, learning_rate=1e-4, **kwargs):
		super().__init__(**kwargs)
		self.representation = Representation()
		self.prediction     = Prediction()
		self.dynamics       = Dynamics()
		self.optimizer      = Adam(learning_rate)


	def call(self, inputs):
		O, A, PI, Z, U = inputs
		s = self.representation(O)
		loss = 0.
		for a, pi, z, u in zip(A, PI, Z, U):
			p, v = self.prediction(s)
			s, r = self.dynamics(s, a)
			policy_loss = -tf.math.reduce_sum(pi * tf.math.log(p), axis=1)
			value_loss  = (z - v)**2
			reward_loss = (u - r)**2
			loss += tf.math.reduce_mean(policy_loss + value_loss + reward_loss)
		return loss


	def train(self, O, A, PI, Z, U):
		with tf.GradientTape() as tape:
			loss = self(O, A, PI, Z, U)
		gradients = tape.gradient(loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))


	def get_config(self):
		base_config = super().get_config()
		return {**base_config}


	def save(self, name):
		self.representation.save_weights(f"{name}_representation.pd")


if __name__ == '__main__':
	model = MuZeroNet()
	#model.load_weights('meep.pd')
	print(model.representation.predict(np.ones((1, 8, 8, 16))).shape)
	#print(model.prediction.predict(np.ones((1, 8, 8, 16))))
	#print(model.dynamics.predict([np.ones((1, 8, 8, 16)), np.ones((1, 8, 8, 4))]))
	#model.save_weights('meep.pd')
	#model.summary()
	quit()
	from time import time


	def timer(f, n):
		t_start = time()
		for _ in range(n):
			f()
		t_stop = time()
		print(f"{(t_stop - t_start)/n}sec")


	def func():
		model.representation.predict(np.ones((128, 8, 8, 16)))
		model.prediction.predict(np.ones((128, 8, 8, 16)))
		model.dynamics.predict([np.ones((128, 8, 8, 16)), np.ones((128, 8, 8, 4))])
	

	timer(func, 10)
	model.representation.summary()
	model.prediction.summary()
	model.dynamics.summary()

	#9999842
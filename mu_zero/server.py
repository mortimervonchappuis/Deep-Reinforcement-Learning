from tqdm import tqdm
from math import log, ceil
import numpy as np
import tensorflow as tf



class NetworkServer:
	def __init__(self, model, connections, replaybuffer):
		self.model        = model
		self.connections  = connections
		self.replaybuffer = replaybuffer
		#self.network      = self.model()
		#print(self.network.representation.predict(np.ones((4, 8, 8, 16))))
		#self.network.prediction.build((None, 8, 8, 16))
		#self.network.dynamics.build(((None, 8, 8, 16), (None, 8, 8, 4)))


	def __getattr__(self, attr):
		if attr == 'network':
			self.network = self.model()
			return self.network


	def __call__(self, timesteps, depth=800, update=1, checkpoint=100, path=None):
		pad = lambda x: str(x).zfill(ceil(log(timesteps, 10)))
		if path is not None:
			self.network.load_weights(f"parameters/{path}.pd")
		with tqdm(total=timesteps * depth) as bar:
			for t in range(timesteps):
				self.represent()
				for k in range(depth):
					self.evaluate()
					bar.update(1)
				if t % update == 0 and len(self.replaybuffer):
					self.train()
				if t % checkpoint == 0:
					self.network.save_weights(f"parameters/{path}_{pad(t)}.pd")


	def represent(self):
		#print('(net)  wait repr/eval')
		O = [con.recv() for con in self.connections]
		#print('(net)  recv repr/eval')
		O = tf.stack(O)
		S = self.network.representation.predict(O)
		P, V  = self.network.prediction.predict(S)
		#print('(net)  start sending repr/eval')
		for s, p, v, con in zip(np.rollaxis(S, axis=0),
								np.rollaxis(P, axis=0),
								np.rollaxis(V, axis=0),
								self.connections):
			con.send([s, p, v])
		#print('(net)  stop  sending repr/eval')


	def evaluate(self):
		#print('(net)  wait eval')
		S_old = tf.stack([con.recv() for con in self.connections])
		#print('(net)  recv eval')
		S, R  = self.network.dynamics(S_old)
		P, V  = self.network.prediction(S)
		#print('(net)  start sending pred/eval')
		for s, r, p, v, con in zip(np.rollaxis(S, axis=0), 
								   np.rollaxis(R, axis=0), 
								   np.rollaxis(P, axis=0), 
								   np.rollaxis(V, axis=0), 
								   self.connections):
			con.send([s, r, p, v])
		#print('(net)  stop  sending pred/eval')


	def train(self):
		if self.replaybuffer:
			batch = self.replaybuffer.fetch()
			self.network.train(batch)

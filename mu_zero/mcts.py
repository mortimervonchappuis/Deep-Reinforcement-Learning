import numpy as np
from math import log, sqrt


class Node:
	def __init__(self, A=None, P=1., S=None, V=0., parent=None, name=None):
		self.parent = parent
		self.leaf   = True
		self.A 	    = A
		self.P 	    = P
		self.S 	    = S
		self.V      = V
		self.R      = 0
		self.N 	    = 0
		self.Q 	    = 0
		self.name   = name


	def __iter__(self):
		for node in self.children:
			yield node


	def __getitem__(self, A):
		return self.children[A]


	def update(self, G):
		self.Q = (self.N * self.Q + G)/(self.N + 1)
		self.N += 1


	def path(self):
		node = self
		while node.parent is not None:
			yield node
			node = node.parent




class MCTS:
	C1 = 1.25
	C2 = 19.652
	def __init__(self, server, O, As, mapping, gamma=1., boardgame=True):
		self.boardgame = boardgame
		self.server    = server
		self.mapping   = mapping
		self.As        = As
		self.Q_max     = float('-inf')
		self.Q_min     = float('inf')
		self.gamma     = gamma
		# INIT ROOT
		#print('(tree) start repr/eval init')
		self.server.send(O)
		S, Ps, V = self.server.recv()
		#print('(tree) stop  repr/eval init')
		self.root = Node(S=S, V=V)
		self.root.children = [Node(A=A, P=P, parent=self.root, name='root') for A, P in zip(self.As, Ps)]
		self.root.leaf = False


	def __call__(self, depth):
		for _ in range(depth):
			node = self.search()
			G = self.expand(node)
			self.backup(node, G)
		return np.array([node.N for node in self.root], dtype=np.float64)


	def search(self):
		node = self.root
		while not node.leaf:
			node = self.select(node)
		return node


	def select(self, node):
		Qs = [n.Q for n in node]
		self.Q_max = max([self.Q_max] + Qs)
		self.Q_min = min([self.Q_min] + Qs)
		return max(node, key=self.UCB)


	def expand(self, node):
		A = self.mapping(node.A)
		#print('(tree) start pred/eval search')
		SA = np.concatenate((node.parent.S, A), axis=2)
		self.server.send(SA)
		#print('(tree) stop  pred/eval search')
		node.S, node.R, Ps, node.V = self.server.recv()
		node.children = [Node(A=A, P=P, parent=node)\
						 for A, P in zip(self.As, Ps)]
		node.leaf = False
		G = 0
		if not self.boardgame and node is not None:
			for n in node.path():
				G = n.R + self.gamma * G
		return G


	def evaluate(self, node):
		node.S = self.con.send([node.parent.S, node.A])
		P, G   = self.con.send(node.S)
		while not self.boardgame or node is not None:
			G = node.R + self.gamma * G
			node = node.parent
		return P, G


	def backup(self, node, G):
		while node is not None:
			node.update(G)
			if not self.boardgame: G *= -1
			node = node.parent


	def UCB(self, node):
		return self.Q_norm(node) + node.P * \
			   (node.parent.N)/sqrt(node.N + 1) * \
			   (self.C1 + log((node.parent.N + \
			   self.C2 + 1)/self.C2))


	def Q_norm(self, node):
		if self.Q_max - self.Q_min != 0:
			return (node.Q - self.Q_min)/(self.Q_max - self.Q_min)
		else:
			#return node.Q
			return node.parent.V

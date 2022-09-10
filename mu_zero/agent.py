from mcts import MCTS
import numpy as np



class Agent:
	def __init__(self, 
				 env, 
				 connection, 
				 gamma, 
				 replaybuffer, 
				 depth,
				 mapping, 
				 temperatur={float('inf'): 1}):
		self.env          = env()
		self.connection   = connection
		self.gamma        = gamma
		self.replaybuffer = replaybuffer
		self.depth        = depth
		self.mapping      = mapping
		self.temperatur   = sorted(temperatur.items(), key=lambda x: x[0])
		self.N            = 0


	def __call__(self, O):
		#print('start repr')
		#S       = self.connection.send(O)
		#print('stop repr')
		tree    = MCTS(self.connection, O, 
					   self.env.action_space, 
					   self.mapping)
		PI      = tree(self.depth)
		mask    = self.env.action_mask
		PI_norm = self.normalize(PI, mask)
		A = np.random.choice(self.env.action_space, p=PI_norm)
		self.N += 1
		return A, PI_norm


	def normalize(self, PI, mask):
		PI_mask = PI * mask
		if PI_mask.sum() == 0:
			return mask/mask.sum()
		for n, T in self.temperatur:
			if n > self.N:
				break
		PI_mask = PI_mask ** (1 / T)
		return PI_mask/PI_mask.sum()


	def run(self, N, PID=42, save=True):
		np.random.seed(PID)
		O = self.env.reset()
		done, trajectory = False, []
		for n in range(N):
			if done:
				O = self.env.reset()
				if save:
					Us_one = [U for _, _, _, u in trajectory[0::2]]
					Us_two = [U for _, _, _, u in trajectory[1::2]]
					# ADDING THE FINAL RESULT TO THE OTHER AGENTS FINAL SCORE
					if len(Us_one) > len(Us_two):
						Us_two.append(U)
					else:
						Us_one.append(U)
					# CONSTRUCTING CUMMULATIVE UTILITY Z (RETURN)
					Z_one  = sum(U * self.gamma**i for i, U in enumerate(Us_one))
					Z_two  = sum(U * self.gamma**i for i, U in enumerate(Us_two))
					Zs = []
					for U_one, U_two in zip(Us_one, Us_two):
						Zs.append(Z_one)
						Zs.append(Z_two)
						Z_one = (Z_one - U_one)/self.gamma
						Z_two = (Z_two - U_two)/self.gamma
					trajectory = [t + (Z,) for t, Z in zip(trajectory, Zs)]
					self.replaybuffer.extend(trajectory)
			A, PI = self(O)
			O_prime, U, done = self.env.step(A)
			if save:
				trajectory.append((O, PI, A, U))
			O = O_prime





if __name__ == '__main__':
	a = Agent(None, None, {10: 1, 100: 0.5, float('inf'): 0.25})
	print(a.temperatur)
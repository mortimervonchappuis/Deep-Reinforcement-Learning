import gym
import random
import os


class GridWORLD(gym.Env):
	metadata = {'render.modes': ['human']}
	WINDY = 0.3
	BUMPING = -3
	DYING = -500
	WINNING = 1000
	DEFAULT = -1
	MAX_N_STEPS = None
	# WIND: West, East, North, South
	# OBJECTS: Lava, Obstacle, Goal
	WORLD = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
			 ['O', 'T', 'T', 'T', 'O', ' ', 'E', 'G', 'W', 'O'],
			 ['O','NE','NE', 'T', 'O', ' ','NE', 'N','NW', 'O'],
			 ['O','NE','NE', 'T', 'O', ' ', ' ', ' ', ' ', 'O'],
			 ['O', ' ', 'O', 'O', 'O', 'O', 'E', ' ', 'T', 'O'],
			 ['O', ' ', ' ', ' ', ' ', 'O', 'E', ' ', 'T', 'O'],
			 ['O','SE', 'S','SW', ' ', 'O', 'E', ' ', 'T', 'O'],
			 ['O', 'E', 'T', 'W', ' ', ' ', ' ', ' ', ' ', 'O'],
			 ['O','NE', 'N','NW', ' ', ' ', ' ', ' ', ' ', 'O'],
			 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]
	WIND = {'N':  [-1,  0], 'E':  [ 0,  1],
			'S':  [ 1,  0], 'W':  [ 0, -1],
			'NW': [-1, -1], 'NE': [-1,  1],
			'SW': [ 1, -1], 'SE': [ 1,  1]}
	#            north     east      south     west
	ACTIONS = [[-1,  0], [ 0,  1], [ 1,  0], [ 0, -1]]
	SYMBOLS = {'N':  ' ↑ ', 'E':  ' → ', 'S':  ' ↓ ', 'W':  ' ← ', 
			   'NW': ' ↖ ', 'NE': ' ↗ ', 'SE': ' ↘ ', 'SW': ' ↙ ', 
			   'O':  ' ■ ', 'G':  ' ★ ', 'T':  ' ☢ ', ' ':  '   ', 'P': ' ✈ '}
	START = (5, 4)


	def __init__(self):
		self.action_space = gym.spaces.Discrete(4)
		self.observation_space = gym.spaces.Discrete(2)
		self.rows = len(self.WORLD)
		self.columns = len(self.WORLD[0])
		self.portals = [(i, j) for i in range(self.rows) for j in range(self.columns) if self.WORLD[i][j] == 'P']


	def step(self, action):
		self.t += 1
		if self.MAX_N_STEPS and self.t  >= self.MAX_N_STEPS:
			self.done = True
			return self.state, 0, self.done, {}
		if self.done:
			return self.state, 0, self.done, {}
		i, j = self.state
		square = self.WORLD[i][j]
		a_di, a_dj = self.ACTIONS[action]
		# CHECK FOR WIND
		if square in self.WIND:
			w_di, w_dj = self.WIND[square]
			# APPLY WIND STOCASTICALLY
			if random.uniform(0, 1) < self.WINDY:
				a_di += w_di
				a_dj += w_dj
				# CLIP MOVEMENT
				if abs(a_di) > 1:
					a_di //= 2
				if abs(a_dj) > 1:
					a_dj //= 2
		# APPLY MOVEMENT
		i_new, j_new = i + a_di, j + a_dj

		match self.WORLD[i_new][j_new]:
			case 'O':
				reward = self.BUMPING
			case 'T':
				reward = self.DYING
				i, j = i_new, j_new
				self.done = True
			case 'G':
				reward = self.WINNING
				i, j = i_new, j_new
				self.done = True
			case 'P':
				reward = self.DEFAULT
				i, j = random.choice(self.portals)
			case _:
				i, j = i_new, j_new
				reward = self.DEFAULT
		self.state = (i, j)
		if not i_new in range(self.rows) or not j_new in range(self.columns):
			reward = self.BUMPING
		return self.state, reward, self.done, {}


	def reset(self):
		self.state = self.START
		self.done = False
		self.t = 0
		return self.state


	def render(self, mode='human'):
		os.system('clear')
		def symbol(i, j):
			if self.WORLD[j][i] != 'O':
				return self.SYMBOLS[self.WORLD[j][i]]
			else:
				left   = i - 1 in range(self.rows) and self.WORLD[j][i-1] == 'O'
				right  = i + 1 in range(self.rows) and self.WORLD[j][i+1] == 'O'
				top    = j - 1 in range(self.columns) and self.WORLD[j-1][i] == 'O'
				bottom = j + 1 in range(self.columns) and self.WORLD[j+1][i] == 'O'
				#          Left  Right   Top  Bottom
				symbs = {( True,  True,  True,  True): '─┼─',
						 ( True,  True,  True, False): '─┴─',
						 ( True,  True, False,  True): '─┬─',
						 ( True,  True, False, False): '───',
						 ( True, False,  True,  True): '─┤ ',
						 ( True, False,  True, False): '─┘ ',
						 ( True, False, False,  True): '─┐ ',
						 ( True, False, False, False): '───',
						 (False,  True,  True,  True): ' ├─',
						 (False,  True,  True, False): ' └─',
						 (False,  True, False,  True): ' ┌─',
						 (False,  True, False, False): '───',
						 (False, False,  True,  True): ' │ ',
						 (False, False,  True, False): ' │ ',
						 (False, False, False,  True): ' │ ',
						 (False, False, False, False): ' ▪ ',}
				return symbs[left, right, top, bottom]


		squares = [[symbol(i, j)
			for i in range(self.rows)] for j in range(self.columns)]
		i, j = self.state
		squares[i][j] = ' × '
		print('\n'.join(''.join(row) for row in squares))

	def seed(self, seed):
		random.seed(seed)



class Labyrinth(GridWORLD):
	WORLD = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
			 ['O', 'T', ' ', ' ', ' ', ' ', ' ', 'O', 'P', 'O', 'T', ' ', ' ', 'O', ' ', 'O', 'G', 'O'],
			 ['O', ' ', 'O', 'O', 'O', 'O', ' ', 'O', ' ', 'O', 'O', 'O', ' ', ' ', ' ', 'O', ' ', 'O'],
			 ['O', ' ', 'O', ' ', ' ', ' ', ' ', 'O', ' ', ' ', ' ', 'O', 'O', 'O', ' ', 'O', ' ', 'O'],
			 ['O', ' ', 'O', ' ', 'O', ' ', 'O', 'O', 'O', ' ', ' ', ' ', 'O', ' ', ' ', ' ', ' ', 'O'],
			 ['O', ' ', ' ', ' ', 'O', ' ', 'O', 'T', 'O', ' ', 'O', ' ', 'O', ' ', 'O', 'O', 'O', 'O'],
			 ['O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', ' ', 'O', ' ', ' ', ' ', 'O', 'P', ' ', 'O'],
			 ['O', ' ', ' ', ' ', 'O', 'P', 'O', ' ', 'O', ' ', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O'],
			 ['O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', ' ', ' ', ' ', 'O', 'O', 'O', ' ', 'O', ' ', 'O'],
			 ['O', ' ', 'O', ' ', ' ', ' ', 'O', ' ', 'O', 'O', ' ', 'O', ' ', ' ', ' ', 'O', ' ', 'O'],
			 ['O', ' ', 'O', 'O', ' ', 'O', 'O', 'O', 'O', 'T', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O'],
			 ['O', ' ', 'O', ' ', ' ', ' ', ' ', 'O', ' ', ' ', ' ', ' ', ' ', 'O', ' ', ' ', ' ', 'O'],
			 ['O', ' ', ' ', ' ', ' ', 'O', ' ', 'O', ' ', 'O', 'O', ' ', 'O', 'O', ' ', 'O', 'O', 'O'],
			 ['O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', ' ', 'O', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'O'],
			 ['O', ' ', ' ', 'O', ' ', ' ', ' ', 'O', ' ', ' ', ' ', 'O', ' ', ' ', ' ', ' ', ' ', 'O'],
			 ['O', ' ', 'O', 'O', ' ', 'O', 'O', 'O', ' ', ' ', ' ', 'O', ' ', 'O', 'O', 'O', ' ', 'O'],
			 ['O', ' ', ' ', 'O', ' ', ' ', ' ', ' ', ' ', 'O', 'P', 'O', ' ', ' ', ' ', ' ', 'T', 'O'],
			 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]
	START = (16, 1)


if __name__ == '__main__':
	env = GridWORLD()
	#env = Labyrinth()
	env.reset()
	env.render()
	d = {'w': 0, 'd': 1, 's': 2, 'a': 3}
	while True:
		try:
			action = d[input('»»»')]
		except:
			break
		O, R, done, info = env.step(action)
		env.render()
		print(R)

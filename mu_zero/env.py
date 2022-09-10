from chess import *
from itertools import product
import numpy as np


board = Board()
#print(board.has_castling_rights(BLACK))
print(sorted(board.__dir__()))
#print(*board.legal_moves)
#print(*board.generate_legal_moves())



#print(bin(board.pieces_mask(PAWN, BLACK)), board.pieces) # color_at
#print(board)
#print(board.push_san('e2e4'))
#print(board)
#print(board.king(WHITE), board.king(BLACK), bin(board.pawns))
#print(board.bishops)

class Chess:
	def __init__(self, state=None):
		self.state = state or Board()
		self.action_space = range(64 * 64 + 256)


	def __call__(self, move):
		state = self.state.copy()
		state.push(move)
		return Chess(state)


	def _getattr__(self, attr):
		if attr.islower():
			raise NotImplemented
		else:
			return getattr(self, attr.lower())


	def __iter__(self):
		for move in self.state.legal_moves:
			yield move


	def step(self, A):
		O = self.x
		A = self.action_map_output_inverse(A)
		done = self.state.is_game_over()
		self.state.push(A)
		winner = None
		if done:
			result = self.state.result().split('-')
			if result[0] == '1':
				winner = WHITE
			elif result[1] == '1':
				winner = BLACK
		if winner is None:
			U = 0
		elif winner == self.state.turn:
			U = -1
		elif winner != self.state.turn:
			U = 1
		return O, U, done


	def reset(self):
		self.state.reset()
		return self.x


	@property
	def x(self):
		# 20.000 per second
		# PIECE TYPES (ROOK, KNIGHT, BISHOP, QUEEN, KING, PAWN)
		# TURN COLOUR
		# CASTLEING WHITE
		# CASTELING BLACK
		# REPETITION?
		x = np.zeros((8, 8, 16))
		colours = [WHITE, BLACK] if self.state.turn == WHITE else [BLACK, WHITE]
		pieces  = [ROOK, KNIGHT, BISHOP, KING, QUEEN, PAWN]
		for i, (colour, piece) in enumerate(product(colours, pieces)):
			x[:, :, i] = np.array(self.__list(piece, colour)).reshape(8, 8)
		x[:, :, 12] = 0 if self.state.turn == WHITE else 1
		x[:, :, 13] = 0 if self.state.has_castling_rights(BLACK if self.state.turn == BLACK else WHITE) else 1
		x[:, :, 14] = 0 if self.state.has_castling_rights(BLACK if self.state.turn == WHITE else WHITE) else 1
		x[:, :, 15] = 0 if self.state.can_claim_threefold_repetition() else 1
		return x


	@property
	def a(self):
		return map(self.action_map_input, self)


	@staticmethod
	def action_map_input(a):
		# SQUARES:   [SOURCE, TARGET]
		# PROMOTION: [KNIGHT, QUEEN]
		a_map = np.zeros((8, 8, 4))
		a_map[a.from_square//8, a.from_square%8, 0] = 1.
		a_map[a.to_square//8,   a.to_square%8,   1] = 1.
		for i, piece in enumerate([KNIGHT, QUEEN]):
			if a.promotion == piece:
				a_map[:, :, i + 2] = 1.
		return a_map


	@staticmethod
	def action_map_output(a):
		if a.promotion is None:
			return a.from_square + a.to_square * 64
		else:
			from_square = a.from_square % 8
			to_square   = a.to_square   % 8
			promotion = (0 if a.promotion == 2 else 1)
			if a.from_square in range(8, 16):
				return promotion * 128 + from_square + to_square * 8
			elif a.from_square in range(48, 56):
				return promotion * 128 + from_square + to_square * 8 + 64
			else:
				raise Exception('ILLEGAl MOVE!')


	@staticmethod
	def action_map_output_inverse(a):
		if a < 64 * 64:
			from_square, to_square = a % 64, a//64
			promotion = None
		else:
			a -= 64 * 64
			from_square =  a % 8
			to_square   = (a % 64)//8 
			rank        =  a < 64
			promotion   =  2 if a//128 == 0 else 5
			from_square = from_square + (8 if rank else 48)
			to_square   = to_square   + (0 if rank else 56)
		return Move(from_square, to_square, promotion)


	@staticmethod
	def action_mapping(a):
		move = Chess.action_map_output_inverse(a)
		return Chess.action_map_input(move)


	@property
	def action_mask(self):
		mask = np.zeros((len(self.action_space)))
		for a in self:
			mask[self.action_map_output(a)] = 1.
		return mask


	def __list(self, piece, colour):
		return list(map(int, bin(board.pieces_mask(piece, colour)).replace('0b', '').zfill(64)))


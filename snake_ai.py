from dqn_agent import DQN_Agent
from snake import SnakeGame, SnakeDirs, Point, dirmap
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np

class SmartSnakeGame(SnakeGame):
	def __init__(self, size):
		super().__init__(size)
		self.make_game_matrix()
		self.snake.change_dir(self.get_safe_dir())

	def make_game_matrix(self):
		m = np.zeros((self.size, self.size))
		p = Point()
		for i in range(self.size):
			for j in range(self.size):
				p.x = j
				p.y = i
				if self.apple == p:
					m[i][j] = 1 # 1 is the apple
				elif p in self.snake.body:
					if p == self.snake.body[0]:
						m[i][j] = 3 # 3 is the head
					else:
						m[i][j] = 2 # 2 is the body

		self.matrix = m.reshape(1, len(m), len(m[0]))

	def pre_update_matrix(self):
		apple = self.apple
		head = self.snake.body[0]
		tail = self.snake.body[len(self.snake.body) - 1]
		
		self.matrix[0][apple.y][apple.x] = 0
		self.matrix[0][head.y][head.x] = 2
		if self.snake.growing == 0:
			self.matrix[0][tail.y][tail.x] = 0

	def post_update_matrix(self):
		apple = self.apple
		head = self.snake.body[0]

		self.matrix[0][apple.y][apple.x] = 1
		self.matrix[0][head.y][head.x] = 3

	def update(self):
		self.pre_update_matrix()
		super().update()
		self.post_update_matrix()

	def get_safe_dir(self):
		head = self.snake.body[0]
		for dir in dirmap.keys():
			if dir != SnakeDirs.none and not self.lose_conditions(head + dirmap[dir]):
				return dir

		
		return SnakeDirs.up

	def turn_snake(self, action):
		if action == 1:
			return

		if action == 0:
			turndir = SnakeDirs.left
		elif action == 2:
			turndir = SnakeDirs.right

		self.snake.change_dir(self.snake.dir.relative_dir(turndir))

class SnakeAgent(DQN_Agent):
	nactions = 3
	def __init__(self, size, each_step = None, network = None):
		self.each_step = each_step
		self.game = SmartSnakeGame(size)
		if network is None:
			q_net = self.make_q_network()
		else:
			q_net = network
			
		super().__init__(q_net)

	def step(self, action):
		state = None
		reward = 0
		
		if callable(self.each_step):
			self.each_step()

		self.game.turn_snake(action)
		self.game.update()

		if self.game.game_over:
			reward += self.game.game_over*0.5

		if self.game.apple_eaten:
			reward += 0.75
		else:
			reward -= 0.05

		if not self.game.game_over:
			state = np.copy(self.game.matrix).reshape(1, -1)

		return state, reward

	def reinit_env(self):
		self.game.__init__(self.game.size)
		return np.copy(self.game.matrix).reshape(1, -1)

	def make_q_network(self):
		inputs = Input(shape = (self.game.size**2, ))
		conv1 = Dense(64, activation = 'elu')(inputs)
		conv2 = Dense(64, activation = 'elu')(conv1)
		out = Dense(3, activation = 'elu', name = 'left')(conv2)
		net = Model(inputs=inputs, outputs=out)
		net.compile(loss='mean_squared_error', optimizer='sgd')
		return net

	def train(self, min_steps = 10000, max_steps = 50000):
		return super().train(np.copy(self.game.matrix).reshape(1, -1), min_steps, max_steps)

	def run(self):
		return super().run(np.copy(self.game.matrix).reshape(1, -1))

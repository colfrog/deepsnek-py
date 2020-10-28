import random
import math
from collections import deque, namedtuple
from tensorflow import keras
import numpy as np

def max_index(array):
	maxval = None
	maxindex = 0
	for i in range(len(array)):
		try:
			test = iter(array[i])
			_, val = max_index(array[i])
		except TypeError:
			val = array[i]

		if maxval is None or val > maxval:
			maxindex, maxval = i, val

	return maxindex, maxval

class EpsilonGreedy():
	def __init__(self, start, end, decay):
		self.epsilon = start
		self.start = start
		self.end = end
		self.decay = decay
		self.steps = 0

	def update(self):
		self.steps += 1
		self.epsilon = self.end + (self.start - self.end)*math.exp(-self.steps*self.decay)

experience = namedtuple('experience', ['state', 'next_state', 'action', 'reward'])
class DQN_Agent():
	memory = deque([])
	def __init__(self, q_net,
		epsilon = EpsilonGreedy(1, 0.1, 0.00005),
		discount_factor = 0.1,
		learning_rate = 1,
		batch_size = 32,
		memory_len = 1000,
                steps_to_target_net_update = 100):
		self.epsilon = epsilon
		self.discount_factor = discount_factor
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.memory_len = memory_len
		self.steps_to_target_net_update = steps_to_target_net_update
		self.q_net = q_net
		self.target_net = keras.models.clone_model(self.q_net)
		self.update_target_network()

	def step(self, action):
		return

	def reinit_env(self):
		return

	def train_q_network(self, states, targets):
		self.q_net.train_on_batch(states, targets)
	
	def update_target_network(self):
		self.target_net.set_weights(self.q_net.get_weights())

	def explore_action(self):
		return random.randrange(0, self.nactions)
	
	def exploit_action(self, state):
		out = self.q_net.predict(state)[0]
		return max_index(out)[0]

	def get_action(self, state):
		n = random.randrange(0, 100)/100
		if n < self.epsilon.epsilon:
			return self.explore_action()
		else:
			return self.exploit_action(state)

	def sort_sample(self, sample):
		states = np.zeros((len(sample), sample[0].state.shape[1]))
		targets = np.zeros((len(sample), 3))

		i = 0
		for e in sample:
			states[i] = e.state[0]

			if e.next_state is None:
				next_max_q = 0
			else:
				next_max_q = np.max(self.target_net.predict(e.next_state))

			target = self.q_net.predict(e.state)[0]
			target[e.action] = e.reward + self.discount_factor*next_max_q
			targets[i] = target

		return states, targets

	def teach_sample(self):
		sample = random.sample(self.memory, self.batch_size)
		states, targets = self.sort_sample(sample)
		self.train_q_network(states, targets)

	def train(self, initial_state, min_steps = 0, max_steps = 15000):
		i = 0
		next_state = initial_state
		self.update_target_network()

		while next_state is not None:
			state = next_state
			action = self.get_action(state)
			next_state, reward = self.step(action)
			e = experience(state, next_state, action, reward)

			if i%self.steps_to_target_net_update:
				self.update_target_network()

			self.memory.append(e)
			if len(self.memory) > self.memory_len:
				self.memory.popleft()
				self.epsilon.update()
				self.teach_sample()

			i += 1
			if i <= min_steps and next_state is None:
				next_state = self.reinit_env()
			if i > max_steps:
				return

	def run(self, initial_state):
		next_state = initial_state
		while next_state is not None:
			state = next_state
			action = self.exploit_action(state)
			next_state, reward = self.step(action)

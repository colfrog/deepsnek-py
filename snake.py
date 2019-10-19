from collections import deque
from enum import Enum, auto
import random

class SnakeDirs(Enum):
	none = 0
	up = 1
	down = 2
	left = 3
	right = 4

	def opposite(self):
		sd = SnakeDirs
		opposites = {
			sd.none: sd.none,
			sd.up: sd.down,
			sd.down: sd.up,
			sd.left: sd.right,
			sd.right: sd.left
		}
		
		return opposites[self]

	def relative_dir(self, to):
		sd = SnakeDirs

		if self == sd.none:
			return to
		if to == sd.none or to == sd.up:
			return self
		if to == sd.down:
			return self.opposite()

		reldir = {
			sd.left: {
				sd.up: sd.left,
				sd.down: sd.right,
				sd.left: sd.down,
				sd.right: sd.up
			}, sd.right: {
				sd.up: sd.right,
				sd.down: sd.left,
				sd.left: sd.up,
				sd.right: sd.down
			}
		}

		return reldir[to][self]

class Point():
	def __init__(self, x = 0, y = 0):
		self.x = x
		self.y = y

	def __eq__(self, p):
		return self.x == p.x and self.y == p.y

	def __ne__(self, p):
		return self.x != p.x or self.y != p.y

	def __add__(self, p):
		return Point(self.x + p.x, self.y + p.y)

	def __iadd__(self, p):
		self.x += p.x
		self.y += p.y
		return self

	def __str__(self):
		return "Point({}, {})".format(self.x, self.y)

	def random(self, max_val):
		self.x = random.randrange(0, max_val)
		self.y = random.randrange(0, max_val)
		return self

class SnakePart(Point):
	def __init__(self, x = 0, y = 0):
		self.x = x
		self.y = y
		self.from_dir = SnakeDirs.none
		self.to_dir = SnakeDirs.none
		
	def __add__(self, p):
		return SnakePart(self.x + p.x, self.y + p.y)

	def __str__(self, p):
		return "SnakePart({}, {}, {}, {})" \
			.format(self.x, self.y, self.from_dir, self.to_dir)

dirmap = {
	SnakeDirs.none: Point(0, 0),
	SnakeDirs.up: Point(0, -1),
	SnakeDirs.down: Point(0, 1),
	SnakeDirs.left: Point(-1, 0),
	SnakeDirs.right: Point(1, 0)
}

class Snake():
	def __init__(self, x = 0, y = 0, dir = SnakeDirs.none, growing = 3):
		head = SnakePart(x, y)
		self.body = deque([head])
		self.dir = dir
		self.growing = growing

	def next_pos(self, dir):
		return self.body[0] + dirmap[dir]
		
	def move(self):
		if self.dir == SnakeDirs.none:
			return

		dir = self.dir
		head = self.body[0]
		head.to_dir = dir

		if self.growing == 0:
			# re-use last element
			self.body.rotate()
			new_head = self.body[0]
			new_head.x = head.x
			new_head.y = head.y
			new_head.to_dir = SnakeDirs.none
		else:
			new_head = SnakePart(head.x, head.y)
			self.body.appendleft(new_head)
			self.growing -= 1

		new_head += dirmap[dir]
		new_head.from_dir = dir.opposite()

	def change_dir(self, dir):
		sd = SnakeDirs
		if self.dir == sd.none or \
		((dir in [sd.up, sd.down]) and (self.dir in [sd.left, sd.right])) or \
		((dir in [sd.left, sd.right]) and (self.dir in [sd.up, sd.down])):
			self.dir = dir
			return True
		return False

def deque_butlast(self):
	new_q = deque(self)
	new_q.pop()
	return new_q
		
class SnakeGame():
	def __init__(self, size):
		self.game_over = False
		self.apple_eaten = False
		self.snake = Snake(size)
		self.snake.body[0].random(size)
		self.size = size
		self.make_apple()

	def make_apple(self):
		self.apple = Point().random(self.size)
		self.place_apple()

	def place_apple(self):
		while self.apple in self.snake.body:
			self.apple.random(self.size)

	def lose_conditions(self, npos):
		return ((self.snake.growing > 0 and npos in self.snake.body) or
			npos in deque_butlast(self.snake.body) or
			npos.x < 0 or
			npos.y < 0 or
			npos.x > self.size - 1 or
			npos.y > self.size - 1)

	def win_conditions(self):
		return len(self.snake.body) == self.size**2

	def is_lost(self):
		if self.snake.dir == SnakeDirs.none:
			return False
		else:
			npos = self.snake.next_pos(self.snake.dir)
			return self.lose_conditions(npos)

	def eat_apple(self):
		if self.apple == self.snake.body[0]:
			self.snake.growing += 1
			self.apple_eaten = True
			self.place_apple()
		else:
			self.apple_eaten = False

	def update(self):
		if self.game_over:
			return

		if self.win_conditions():
			self.game_over = 1
			return 1
		if self.is_lost():
			self.game_over = -1
			return -1

		self.eat_apple()
		self.snake.move()

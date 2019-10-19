import os
import sys
import sdl2
import sdl2.ext
import argparse
from snake import SnakeGame, SnakeDirs
from snake_ai import SnakeAgent
from collections import deque
from threading import Thread, Lock
from tensorflow.keras.models import model_from_json

cell_size = 20
game_lock = Lock()

def draw_point(p, renderer, r, g, b, a):
	global cell_size
	rect = sdl2.SDL_Rect(p.x*cell_size, p.y*cell_size, cell_size, cell_size)
	sdl2.SDL_SetRenderDrawColor(renderer, r, g, b, a)
	sdl2.SDL_RenderDrawRect(renderer, rect)

def draw_points(points, renderer, r, g, b, a):
	for p in points:
		draw_point(p, renderer, r, g, b, a)

def draw_board(wsize, renderer):
	rect = sdl2.SDL_Rect(0, 0, wsize, wsize)
	sdl2.SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0)
	sdl2.SDL_RenderFillRect(renderer, rect)

def make_custom_rect(x, y, params):
	return sdl2.SDL_Rect(x + params[0], y + params[1], params[2], params[3])

def get_dir_rects():
	global cell_size
	c = cell_size # 1 cell
	hc = int(cell_size/2) # 1 half cell
	qc = int(cell_size/4) # 1 quarter cell

	sd = SnakeDirs
	return {
		sd.none: [qc, qc, hc, hc],
		sd.up: [qc, 0, hc, 3*qc],
		sd.down: [qc, qc, hc, 3*qc],
		sd.left: [0, qc, 3*qc, hc],
		sd.right: [qc, qc, 3*qc, hc]
	}

def draw_snake(game, renderer):
	global cell_size
	sdl2.SDL_SetRenderDrawColor(renderer, 0, 255, 0, 0)
	dir_rects = get_dir_rects()
	body = deque(game.snake.body)
	for part in body:
		x = cell_size*part.x
		y = cell_size*part.y
		from_rect = make_custom_rect(x, y, dir_rects[part.from_dir])
		to_rect = make_custom_rect(x, y, dir_rects[part.to_dir])

		sdl2.SDL_RenderFillRect(renderer, from_rect)
		sdl2.SDL_RenderFillRect(renderer, to_rect)

def draw_apple(game, renderer):
	global cell_size
	x = int(game.apple.x*cell_size + cell_size/4)
	y = int(game.apple.y*cell_size + cell_size/4)
	size = int(cell_size/2)
	rect = sdl2.SDL_Rect(x, y, size, size)
	sdl2.SDL_SetRenderDrawColor(renderer, 255, 0, 0, 0)
	sdl2.SDL_RenderFillRect(renderer, rect)

def render_game(wsize, game, renderer):
	draw_board(wsize, renderer)
	draw_apple(game, renderer)
	draw_snake(game, renderer)
	sdl2.SDL_RenderPresent(renderer)

def game_loop(game, delay_time):
	while game.game_over == False:
		game_lock.acquire()
		game.update()
		game_lock.release()
		sdl2.SDL_Delay(delay_time)

def handle_keypress(game, sym):
	if sym == sdl2.SDLK_q:
		sys.exit()

	sd = SnakeDirs
	targets = {
		sdl2.SDLK_UP: sd.up,
		sdl2.SDLK_w: sd.up,
		sdl2.SDLK_DOWN: sd.down,
		sdl2.SDLK_s: sd.down,
		sdl2.SDLK_LEFT: sd.left,
		sdl2.SDLK_a: sd.left,
		sdl2.SDLK_RIGHT: sd.right,
		sdl2.SDLK_d: sd.right
	}

	if sym in targets.keys() and game.snake.change_dir(targets[sym]):
		return True

	return False

def show_game(game, r, agent = None, interactive = True):
	global cell_size
	events = sdl2.ext.get_events()
	for e in events:
		if e.type == sdl2.SDL_QUIT:
			sys.exit()
		elif e.type == sdl2.SDL_KEYDOWN:
			sym = e.key.keysym.sym
			if interactive and handle_keypress(game, sym):
				game_lock.acquire()
				game.update()
				sdl2.SDL_Delay(50)
				game_lock.release()
			if agent is not None and sym == sdl2.SDLK_SPACE:
				if agent.delay != 0:
					agent.delay = 0
				else:
					agent.delay = 250

	if agent is not None:
		sdl2.SDL_Delay(agent.delay)
	else:
		sdl2.SDL_Delay(10)

	render_game(cell_size*game.size, game, r)

def init_sdl2(game, window_name):
	global cell_size
	sdl2.ext.init()
	wsize = cell_size*game.size
	window = sdl2.SDL_CreateWindow(bytes(window_name, 'utf-8'), sdl2.SDL_WINDOWPOS_UNDEFINED, sdl2.SDL_WINDOWPOS_UNDEFINED, wsize, wsize, sdl2.SDL_WINDOW_SHOWN)
	renderer = sdl2.SDL_CreateRenderer(window, -1, sdl2.SDL_RENDERER_ACCELERATED)
	return window, renderer

def start_game(board_size, delay_time = 250):
	game = SnakeGame(board_size)
	win, ren = init_sdl2(game, 'snek')

	game_thread = Thread(target=game_loop, args=(game, delay_time))
	game_thread.start()	
	while game_thread.is_alive():
		show_game(game, ren)

def watch_ai(board_size = 15, agent = None):
	if agent is None:
		agent = SnakeAgent(board_size)

	agent.delay = 250
	win, ren = init_sdl2(agent.game, 'smart snek')
	agent.each_step = lambda: show_game(agent.game, ren, agent, False)
	
	agent.run()

def save_model(agent, save_file):
	with open(save_file+'.json', 'w') as json_file:
		json = agent.q_net.to_json()
		json_file.write(json)
	agent.q_net.save_weights(save_file+'.h5')

def load_model(agent, save_file):
	with open(save_file+'.json', 'r') as json_file:
		json = json_file.read()
		agent.q_net = model_from_json(json)
	agent.q_net.load_weights(save_file+'.h5')

parser = argparse.ArgumentParser(prog = 'deepsnek')
parser.add_argument('-t', dest = 'train', action = 'store_true',
		help = 'Train the agent')
parser.add_argument('-s', dest = 'save_file', action = 'store', required = True,
		help = 'The file to save the model in')
parser.add_argument('--min-steps', dest = 'min_steps', action = 'store', type=int,
		help = 'The minimum amount of steps to train for')
parser.add_argument('--max-steps', dest = 'max_steps', action = 'store', type=int,
		help = 'The maximum amount of steps to train for')
args = parser.parse_args()

agent = SnakeAgent(15)
if os.path.exists(args.save_file+'.json') and os.path.exists(args.save_file+'.h5'):
	load_model(agent, args.save_file)

if args.train:
	agent.train(args.min_steps or 10000, args.max_steps or 50000)
else:
	watch_ai(agent = agent)

if args.save_file is not None:
	save_model(agent, args.save_file)

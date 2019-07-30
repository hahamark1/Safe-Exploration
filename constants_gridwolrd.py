MOVES = ['moveLeft', 'moveRight', 'moveUp', 'moveDown']
num_actions = len(MOVES)

# from models import *
from collections import namedtuple

# NETWORK = QNetwork
NUM_EPISODES = 10000
MEMORY_SIZE = 10000
SEED = 42
DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
SAVE_EVERY = 2000
PLOT_EVERY = 500
PRINT_EVERY = 1000
EMBEDDING_SIZE = 4

EMBEDDING = False
DYNAMIC_START = False
DYNAMIC_END = False
DYNAMIC_HOLES = False



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# model_dict = {'QNetwork': }
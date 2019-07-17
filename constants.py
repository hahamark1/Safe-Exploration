EXPERIMENT = 'SE'

VALID_ACTIONS = [0, 1, 2, 3,4 ,5 ]
WINDOW_LENGTH = 4
IMAGE_SIZE = 32
CONV_1 = 4
CONV_2 = 8
CONV_3 = 16
MAX_PIXEL = 255.0

EMBEDDINGS = {'Mario': 1,
              'Other_agents': 2,
              'Ground': 3,
              'Hole': 4,
              'Animal': 5,
              'Coin': 6,
              'Random_box': 7,
              'Random_box_trig': 8
}
EMBEDDING_SIZE = len(EMBEDDINGS.keys()) + 1

EXPERIMENT_NAME = 'safe_exploration_7.4'
HEADLESS = True
LEVEL_NAME = 'Level-basic-one-hole-three-coins.json'
# LEVEL_NAME = 'Level-5-coins.json'
ER_SIZE = 100000
PARTIAL_OBSERVATION = False
DISTANCE_REWARD = True
POLICY = 'GREEDY'

EPISODE_LENGTH = 500
STEP_SIZE = 2
STEP_SIZE_VIDEO = 1

HOLE_REWARD = 1
COIN_REWARD = 1
BLOCK_SIZE = 29.8

MOVES = ['moveLeft', 'moveRight', 'jump', 'jumpLeft', 'jumpRight', 'doNothing']
MAP_MULTIPLIER = 30.9
MAP_HEIGHT = 64
MAP_WIDTH = 76

REPLAY_MEMORY_SIZE = 100000
EPSILON_DECAY_STEPS = 1000000
USE_MEMORY = False
PRIORITIZE_MEMORY = False

epsilon = 0.01
SALIENCY = True
MIN_EPSILON = 0.01

LR = 0.025
WEIGHT_DECAY = 0.99
MOMENTUM = 0.0
Epsilon_network = 1e-6

DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 32
UPDATE_TARGET_STEP = 1000

SECOND_AGENT_ACTION = 'RANDOM' # in ['RANDOM', 'SAME_NETWORK']

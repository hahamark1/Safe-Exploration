from MarioGym import MarioGym
import matplotlib.pyplot as plt
import random
LEVEL_NAME = 'Level-basic-one-hole-three-coins.json'

# Get the environment and extract the number of actions.
env = MarioGym(headless=False, level_name=LEVEL_NAME, no_coins=5, partial_observation=False)

restart = False
plt.ion()

map_length = env.level.levelLength
counter = 0
while True:

    counter += 1
    if restart:
        env.reset(levelname=LEVEL_NAME)

    obs, reward, restart, info = env.step(1)

    counter += 0
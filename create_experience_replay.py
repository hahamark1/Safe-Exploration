from MarioGym import MarioGym, pygame
import matplotlib.pyplot as plt
import pygame
pygame.init()
import random
import numpy as np
import pickle
from DQN_SE import StateProcessor
import tensorflow as tf
from collections import deque, namedtuple

from constants import *


def get_input():
    if pygame.key.get_focused():
        pygame.event.pump()
        events = pygame.key.get_pressed()
        if events[pygame.K_UP]:
            move = 2
            if events[pygame.K_LEFT]:
                move = 3
            if events[pygame.K_RIGHT]:
                move = 4
        else:
            if events[pygame.K_LEFT]:
                move = 0
            elif events[pygame.K_RIGHT]:
                move = 1
            else:
                move = 5
    else:
        move = 5
    return move

def save_replay_memory(memory):

    file_name = '{}_{}.pt'.format(LEVEL_NAME, len(memory))
    with open(file_name, 'wb') as pf:
        pickle.dump(memory, pf)

tf.reset_default_graph()
# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)


# Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

replay_memory = []

# State processor
state_processor = StateProcessor()

replay_memory_size = 100

# Get the environment and extract the number of actions.
env = MarioGym(HEADLESS, step_size=10, level_name=LEVEL_NAME, partial_observation=PARTIAL_OBSERVATION, distance_reward=DISTANCE_REWARD)

done = False
plt.ion()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    total_state = env.reset(levelname=LEVEL_NAME)
    state = state_processor.process(sess, total_state, 1)
    state = np.stack([state] * WINDOW_LENGTH, axis=0)
    total_state = np.stack([state], axis=0)

    map_length = env.level.levelLength

    counter = 0

    while counter < replay_memory_size:
        if done:
            total_state = env.reset(levelname=LEVEL_NAME)

        action = get_input()

        next_total_state, reward, done, info = env.step(action)

        next_state = state_processor.process(sess, next_total_state, 1)
        next_state = np.append(state[1:, :, :], np.expand_dims(next_state, 0), axis=0)

        next_total_state = np.stack([next_state], axis=0)

        replay_memory.append([total_state, action, reward, next_total_state, done])

        counter += 1

        total_state = next_total_state

save_replay_memory(replay_memory)

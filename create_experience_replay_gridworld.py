from GridworldGym import GridworldGym
import matplotlib.pyplot as plt
import pygame
pygame.init()
import random
import numpy as np
import time
import pickle
from joblib import Parallel, delayed
from Q_network_gridworld import ReplayMemory

from constants import *


def get_input():
    move = False
    while not move:
        if pygame.key.get_focused():
            pygame.event.pump()
            events = pygame.key.get_pressed()
            if events[pygame.K_UP]:
                move = 1
            elif events[pygame.K_DOWN]:
                move = 5
            elif events[pygame.K_LEFT]:
                move = 3
            elif events[pygame.K_RIGHT]:
                move = 2
    if move == 5:
        move = 0
    return move

memory_folder = 'memories'


def save_replay_memory(memory, gw_size, supervision, hole_pos=False, Mark=True):
    if hole_pos:
        file_name = '{}/{}_{}_gw={}_sup={}_Mark={}_hp={}.pt'.format(memory_folder, 'GW', len(memory), gw_size, supervision, Mark, hole_pos)
    else:
        file_name = '{}/{}_{}_gw={}_sup={}_Mark={}.pt'.format(memory_folder, 'GW', len(memory), gw_size, supervision, Mark)
    with open(file_name, 'wb') as pf:
        pickle.dump(memory, pf)


# Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

replay_memory_size = 1000

replay_memory = ReplayMemory(replay_memory_size)


def main(gridworld_size, supervision, specific_holes):


    change = True if specific_holes else False
    dynamic_holes = True
    # holes = [[5,1],[1,2],[3,3],[2,4],[5,5]]
    use_holes = False
    # supervision = False
    Mark = False

    # Get the environment and extract the number of actions.
    env = GridworldGym(dynamic_holes=dynamic_holes,constant_change=change, gridworld_size=gridworld_size, specific_holes=specific_holes, self_play=False)


    done = False


    s = env.reset()

    counter = 0

    old_states = [0,0,s]
    while counter < replay_memory_size:
        if done:
            s = env.reset()

        if Mark:
            a = get_input()
        else:
            a = env.optimal_choice()


        s_next, r, done, _ = env.step(a)

        old_states = [old_states[1], old_states[2], s_next]
        if old_states[0] is old_states[2]:
            done = True
        if supervision:
            opt_a = env.optimal_choice()

        if not supervision:
            replay_memory.push((s, a, r, s_next, done))
        else:
            replay_memory.push((s, a, opt_a, r, s_next, done))
        if Mark:
            time.sleep(0.3)
        counter +=1

    if change:
        save_replay_memory(replay_memory, gridworld_size, supervision, hole_pos=specific_holes, Mark=Mark)
    else:
        save_replay_memory(replay_memory, gridworld_size, supervision, Mark=Mark)



#
gridworld_sizes = [7, 15, 24]
supervision = [True, False]
holes = [[[5,1],[1,2],[3,3],[2,4],[5,5]], False]

Parallel(n_jobs=4)(
    delayed(main)(gw, sup, hol) for gw in gridworld_sizes for sup in supervision for hol in holes)


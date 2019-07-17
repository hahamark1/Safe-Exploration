import pygame
import random
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import gym
from gym import spaces
import math
from gym.utils import seeding

MOVES = ['moveLeft', 'moveRight', 'moveUp', 'moveDown']
PLOT = False
GRIDWORLD_SIZE = 7
MAX_STEPS = 400
START_POS = [0,0]
END_POS = [6,6]
HOLE_POS = [[2,1],[3,3],[4,5]]
NUM_HOLES = 3
EMBEDDING_SIZE = 4

class GridworldGym(gym.Env):

    def __init__(self, headless=True, dynamic_start=False, dynamic_holes=False, dynamic_end=False, embedding=False, gridworld_size=7, constant_change=False):
        self.headless = headless
        self.dynamic_start = dynamic_start
        self.dynamic_end = dynamic_end
        self.dynamic_holes = dynamic_holes
        self.action_space = spaces.Discrete(4)
        self.gridworld_size = gridworld_size
        self.num_holes = gridworld_size - 2

        if not self.headless:
            self.fig = plt.figure()
            plt.ion()

        self.change = constant_change
        self.embedding = embedding
        self.observation_space = spaces.Box(low=-10000000, high=100000000, dtype=np.float, shape=(7, 7, 2))

        self.hole_pos = False
        self.reset()

    def reset(self):

        taken_spots = []

        if not self.dynamic_start:
            self.agent_position = START_POS
        else:
            if not self.dynamic_holes:
                self.agent_position = HOLE_POS[0]
                while self.agent_position in HOLE_POS:
                    self.agent_position = [random.randint(0,6), random.randint(0,6)]
            else:
                self.agent_position = [random.randint(0, 6), random.randint(0, 6)]

        taken_spots.append(self.agent_position)
        if not self.dynamic_end:
            self.end_position = END_POS
        else:
            self.end_position = self.agent_position
            while self.end_position in taken_spots:
                self.end_position = [random.randint(0,6), random.randint(0,6)]
        taken_spots.append(self.end_position)

        if not self.dynamic_holes:
            self.hole_pos = HOLE_POS

        else:
            if not self.hole_pos or self.change:
                self.hole_pos = []
                for i in range(1, self.gridworld_size-1):
                    position = [i, random.randint(0,6)]
                    taken_spots.append(position)
                    self.hole_pos.append(position)
        self.steps = 0
        self.get_observation()
        return self.observation

    def get_observation(self):
        if not self.embedding:
            self.observation = np.zeros((self.gridworld_size, self.gridworld_size))
            self.observation[self.end_position[0], self.end_position[1]] = 2
            self.observation[self.agent_position[0], self.agent_position[1]] = 1
            for hole in self.hole_pos:
                self.observation[hole[0], hole[1]] = -1
        else:
            self.observation = np.zeros((EMBEDDING_SIZE, self.gridworld_size, self.gridworld_size))
            self.observation[3, self.end_position[0], self.end_position[1]] = 1
            self.observation[2, self.agent_position[0], self.agent_position[1]] = 1
            for hole in self.hole_pos:
                self.observation[1, hole[0], hole[1]] = -1

        return self.observation

    def plot_env(self):

        self.observation = self.get_observation()

        if not self.headless and not self.embedding:
            plt.clf()
            plt.matshow(self.observation)
            plt.draw()
            plt.pause(0.1)

    def step(self, action_num):

        #increase counter
        self.steps += 1

        #update agents
        restart = (self.agent_position in self.hole_pos or self.agent_position == self.end_position)

        if not restart:

            #move agent
            if action_num == 0:
                self.agent_position = [(self.agent_position[0] + 1) , self.agent_position[1]]
            elif action_num == 1:
                self.agent_position = [(self.agent_position[0] - 1), self.agent_position[1]]
            elif action_num == 2:
                self.agent_position = [self.agent_position[0], (self.agent_position[1] + 1)]
            elif action_num == 3:
                self.agent_position = [self.agent_position[0], (self.agent_position[1] - 1)]


            if self.agent_position[0] >= self.gridworld_size:
                self.agent_position[0] = 6
            elif self.agent_position[0] < 0:
                self.agent_position[0] = 0
            if self.agent_position[1] >= self.gridworld_size:
                self.agent_position[1] = 6
            elif self.agent_position[1] < 0:
                self.agent_position[1] = 0

            if not self.headless:
                self.plot_env()

        info = {}

        reward = 0

        if self.agent_position == self.end_position:
            reward += 1
            restart = True
            info['succeed'] = 1
        elif self.agent_position in self.hole_pos:
            reward -= 1
            info['death'] = 1
            restart = True

        restart = restart or self.steps > MAX_STEPS

        # if restart:
        #     self.reset()
        #     if not self.headless:
        #         plt.pause(3.0)

        self.observation = self.get_observation()

        return self.observation, reward, restart, info

    def render(self, mode='human', close=False):
        pass

    def safest_action(self, observation):
        next_positions = []
        action_scores = np.zeros(self.action_space)



        for action_num in range(self.action_space):
            if action_num == 0:
                next_positions.append([(agent_position[0] + 1) , agent_position[1]])
            elif action_num == 1:
                next_positions.append([(agent_position[0] - 1), agent_position[1]])
            elif action_num == 2:
                next_positions.append([agent_position[0], (agent_position[1] + 1)])
            elif action_num == 3:
                next_positions.append([agent_position[0], (agent_position[1] - 1)])




1
if __name__ == "__main__":
    env = GridworldGym(headless=False)


    while True:
        action = np.random.choice(range(4))
        env.step(action)

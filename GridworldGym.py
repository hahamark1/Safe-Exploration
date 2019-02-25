import pygame
import random
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym.utils import seeding

MOVES = ['moveLeft', 'moveRight', 'moveUp', 'moveDown']


class GridworldGym(gym.Env):

    def __init__(self, headless=True):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-10000000, high=100000000, dtype=np.float, shape=(40, 40))
        plt.ion()

        self.reset()

    def reset(self):

        self.agent_position = [20, 20]
        self.enemy_positions = [[x, y] for x in range(5, 40, 5) for y in range(5, 40, 5)]
        self.score = 0

        self.get_observation()

        return self.observation

    def get_observation(self):
        self.observation = np.zeros((40, 40))

        self.observation[max(0, min(self.agent_position[0], 40-1)), max(0, min(self.agent_position[1], 40-1))] = 1
        for pos in self.enemy_positions:
            self.observation[max(0, min(pos[0], 40-1)), max(0, min(pos[1], 40-1))] = -1
        return self.observation

    def step(self, action_num):

        #move enemies
        for i, pos in enumerate(self.enemy_positions):
            action = np.random.choice(4)
            if action == 0:
                pos = [(pos[0] + 1) % 40, pos[1]]
            elif action == 1:
                pos = [(pos[0] - 1) % 40, pos[1]]
            elif action == 2:
                pos = [pos[0], (pos[1] + 1) % 40]
            elif action == 3:
                pos = [pos[0], (pos[1] - 1) % 40]
            self.enemy_positions[i] = pos

        #kill agent
        restart = self.agent_position in self.enemy_positions
        print(self.agent_position)
        print(restart)
        #print(self.enemy_positions)

        self.observation = self.get_observation()

        if True:
            plt.matshow(self.observation, 0)
            plt.draw()
            plt.pause(0.0001)
            plt.clf()

        #move agent
        if action_num == 0:
            self.agent_position = [(self.agent_position[0] + 1) % 40, self.agent_position[1]]
        elif action_num == 1:
            self.agent_position = [(self.agent_position[0] - 1) % 40, self.agent_position[1]]
        elif action_num == 2:
            self.agent_position = [self.agent_position[0], (self.agent_position[1] + 1) % 40]
        elif action_num == 3:
            self.agent_position = [self.agent_position[0], (self.agent_position[1] - 1) % 40]

        #kill enemies
        if self.agent_position in self.enemy_positions:
            self.enemy_positions = [x for x in self.enemy_positions if x != self.agent_position]

        self.observation = self.get_observation()

        if True:
            plt.matshow(self.observation, 0)
            plt.draw()
            plt.pause(0.0001)
            plt.clf()

        reward = 1
        info = {'info': 'jeej'}


        return self.observation, reward, restart, info

    def render(self, mode='human', close=False):
        pass


if __name__ == "__main__":
    env = GridworldGym()


    while True:
        action = np.random.choice(4)
        env.step(action)
        plt.matshow(env.observation, 0)
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

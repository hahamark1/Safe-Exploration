import pygame
import random
import pickle
import numpy as np
import time
from classes.Level import Level
from entities.Mario import Mario
from classes.Dashboard import Dashboard
from classes.Sound import Sound
from classes.Menu import Menu
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt

MOVES = ['moveLeft', 'moveRight', 'jump', 'jumpLeft', 'jumpRight', 'doNothing']


class MarioGym(gym.Env):

    def __init__(self):
        self.levelname = 'Level-basic-with-goombas.json'
        self.init_game()

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-10000000, high=100000000, dtype=np.float, shape=(40,80,4))
        self.reset()

    def reset(self):
        self.init_game()
        self.observation = self.level_to_numpy()
        return self.observation

    def step(self, action_num):
        action = MOVES[action_num]
        reward = self.do_game_step(action)
        self.observation = self.level_to_numpy()
        if self.observation.shape != (40, 80):
            self.observation = np.zeros((40, 80))

        info = {'info': 'jeej'}
        restart = len([x for x in self.level.entityList if (x.__class__.__name__ == 'Goomba' and x.alive)]) != 11 or self.mario.restart
        #restart = False
        return self.observation, reward, restart, info

    def render(self, mode='human', close=False):
        pass

    def init_game(self):
        self.level = Level(self.levelname)
        self.mario = Mario(0, 0, self.level)
        self.clock = 0


    def do_game_step(self, move):
        start_score = self.level.points
        deadbonus = 0
        mario_position = (int(self.mario.rect.y / 8), int(self.mario.rect.x / 8))
        counter = 0
        while mario_position == (int(self.mario.rect.y / 8), int(self.mario.rect.x / 8)) and counter < 1:
            counter += 1
            self.do_move(move)
            self.mario.update()
            self.level.updateEntities()
            self.clock += (1.0/60.0)

            if self.mario.restart:
                deadbonus = -10000
                break

        reward = 1

        return reward

    def level_to_numpy(self):
        array = np.zeros((600, 600))
        padding = 40
        array[int(round(self.mario.rect.y/8)) -1][int(round(self.mario.rect.x/8)) -1] = -1
        #for i, row in enumerate(self.level.level):
        #    for j, ele in enumerate(row):
        #        if ele.rect:
        #            array[i][j] = 1
        for entity in self.level.entityList:
            if entity.__class__.__name__ == 'Coin':
                array[int(round(entity.rect.y / 8))][int(round(entity.rect.x / 8))] = 2
            if entity.__class__.__name__ == 'Koopa' or entity.__class__.__name__ == 'Goomba':
                    array[int(round(entity.rect.y / 8)) -1][int(round(entity.rect.x / 8)) -1] = 1
            if entity.__class__.__name__ == 'RandomBox':
                if not entity.triggered:
                    array[int(round(entity.rect.y / 8))][int(round(entity.rect.x / 8))] = 4
                else:
                    array[int(round(entity.rect.y / 8))][int(round(entity.rect.x / 8))] = 5
        array = np.hstack((np.zeros((600, padding-5)), array))
        array = np.hstack((np.ones((600, 5)), array))
        return array[12:52,
               int(round(self.mario.rect.x / 8)):int(round(self.mario.rect.x / 8)) + 2 * padding]

    def level_to_full_numpy(self):
        array = np.zeros((700, 700))
        padding = 40
        array[int(round(self.mario.rect.y/8))][int(round(self.mario.rect.x/8))] = -2
        #for i, row in enumerate(self.level.level):
        #    for j, ele in enumerate(row):
        #        if ele.rect:
        #            array[i][j] = 1
        for entity in self.level.entityList:
            if entity.__class__.__name__ == 'Coin':
                array[int(round(entity.rect.y / 8))][int(round(entity.rect.x / 8))] = 2
            if entity.__class__.__name__ == 'Koopa' or entity.__class__.__name__ == 'Goomba':
                if entity.getPosIndex().y < 600:
                    array[int(round(entity.rect.y / 8))][int(round(entity.rect.x / 8))] = 3
            if entity.__class__.__name__ == 'RandomBox':
                if not entity.triggered:
                    array[int(round(entity.rect.y / 8))][int(round(entity.rect.x / 8))] = 4
                else:
                    array[int(round(entity.rect.y / 8))][int(round(entity.rect.x / 8))] = 5
        #array = np.hstack((np.zeros((600, padding-5)), array))
        #array = np.hstack((np.ones((600, 5)), array))
        return array

    def level_to_supersimple_numpy(self):
        array = np.zeros(6)
        min_distance = 1000000
        for entity in self.level.entityList:
            if entity.__class__.__name__ == 'Goomba':
                distance = np.absolute(self.mario.rect.x - entity.rect.x)
                if distance < min_distance:
                    min_distance = distance
                    array[0] = self.mario.rect.x - entity.rect.x
                    array[1] = self.mario.rect.y - entity.rect.y
                    array[5] = entity.vel.x
        array[2] = self.mario.vel.x
        array[3] = self.mario.vel.y
        array[4] = self.mario.rect.x
        return array

        

    def do_move(self, move):
        if move == 'moveLeft':
            self.mario.traits['goTrait'].direction = -1
        elif move == 'moveRight':
            self.mario.traits['goTrait'].direction = 1
        elif move == 'jump':
            self.mario.traits['jumpTrait'].start()
        elif move == 'jumpRight':
            self.mario.traits['goTrait'].direction = 1
            self.mario.traits['jumpTrait'].start()
        elif move == 'jumpLeft':
            self.mario.traits['goTrait'].direction = -1
            self.mario.traits['jumpTrait'].start()
        elif move == 'doNothing':
            self.mario.traits['goTrait'].direction = 0



if __name__ == "__main__":
    env = MarioGym()


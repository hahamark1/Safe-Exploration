import pygame
import random
import pickle
import numpy as np
import time
from classes.Level import Level
from classes.LevelHeadless import LevelHeadless
from entities.Mario import Mario
from entities.Coin import Coin
from entities.CoinHeadless import CoinHeadless

from entities.MarioHeadless import MarioHeadless
from classes.Dashboard import Dashboard
from classes.Sound import Sound
from classes.Menu import Menu
import gym
from gym import spaces
import matplotlib.pyplot as plt
from gym.utils import seeding

EPISODE_LENGTH = 800
HOLE_REWARD = 5
COIN_REWARD = 1
MOVES = ['moveLeft', 'moveRight', 'jump', 'jumpLeft', 'jumpRight', 'doNothing']
MAP_MULTIPLIER = 30.9

class MarioGym(gym.Env):

    def __init__(self, headless=True, level_name='Level-basic-one-goomba.json', no_coins=5):
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-10000000, high=100000000, dtype=np.float, shape=(40, 80, 4))

        self.levelname = level_name
        self.headless = headless
        self.score = 0
        self.max_frame_rate = 60
        self.steps = 0
        self.observation = None
        self.level = None
        self.mario = None
        self.screen = None
        self.dashboard = None
        self.sound = None
        self.menu = None
        self.no_coins = no_coins
        self.coins_start = self.no_coins

        self.init_game()
        self.reset(levelname = self.levelname)

    def reset(self, levelname=None):
        if not levelname:
            self.no_coins = 5
            self.levelname = 'Level-{}-coins.json'.format(self.no_coins)
        else:
            self.levelname = levelname
        self.init_game()
        self.steps = 0
        self.observation = self.level_to_empathic_numpy()
        return self.observation

    def reset_clean(self, y_pos):
        self.count_entities()
        self.coins_taken = self.coins_start - self.no_coins
        self.coins_end = self.no_coins
        self.no_coins = min(5, self.no_coins * 2)
        self.coins_start = self.no_coins

        self.levelname = 'Level-{}-coins.json'.format(self.no_coins)
        self.init_game(y_position=y_pos, coins=self.coins_end, clock=self.clock)

        self.observation = self.level_to_empathic_numpy()

        return self.observation

    def step(self, action_num):
        self.steps += 1
        action = MOVES[action_num]
        num_goombas= len([x for x in self.level.entityList
                          if ((x.__class__.__name__ == 'Goomba'
                              or x.__class__.__name__ == 'GoombaHeadless')
                              and x.alive)])

        coins = len([x for x in self.level.entityList
                          if ((x.__class__.__name__ == 'Coin'))])

        old_x_pos = self.mario.rect.x / (10 * MAP_MULTIPLIER)

        reward = self.do_game_step(action)
        coins_taken = reward/COIN_REWARD

        goombas_died = num_goombas - len([x for x in self.level.entityList
                          if ((x.__class__.__name__ == 'Goomba'
                              or x.__class__.__name__ == 'GoombaHeadless')
                              and x.alive)])

        coins_taken = coins - len([x for x in self.level.entityList if ((x.__class__.__name__ == 'Coin'))])
        self.observation = self.level_to_empathic_numpy()
        # print(coins_taken)
        info = {'num_killed': goombas_died,
                'coins_taken': coins_taken,
                'death': self.mario.restart}


        if self.mario.restart:
            reward -= HOLE_REWARD
        if coins_taken > 0:
            self.mario.restart = True

        restart = self.mario.restart or self.steps >= EPISODE_LENGTH
        return self.observation, reward, restart, info

    def render(self, mode='human', close=False):
        pass

    def init_game(self, y_position=0, coins=0, points=0, time=0, clock=0):
        if not self.headless:
            pygame.mixer.pre_init(44100, -16, 2, 4096)
            pygame.init()
            self.screen = pygame.display.set_mode((640, 480))
            self.dashboard = Dashboard("./img/font.png", 8, self.screen, coins=0, points=0, time=0)
            self.sound = Sound()
            self.level = Level(self.screen, self.sound, self.dashboard, self.levelname)
            self.menu = Menu(self.screen, self.dashboard, self.level, self.sound)
            self.menu.update()

            self.mario = Mario(0, y_position/32, self.level, self.screen, self.dashboard, self.sound)
            self.clock = pygame.time.Clock()

            self.menu.dashboard.state = "play"
            self.menu.dashboard.time = 0
            self.menu.start = True

            pygame.display.update()
        else:
            self.level = LevelHeadless(self.levelname)
            self.mario = MarioHeadless(0, 0, self.level)
            self.clock = clock

    def do_game_step(self, move):
        if not self.headless:
            start_score = self.dashboard.points
        else:
            start_score = self.level.points

        counter = 0
        reward = 0
        while counter < 5:
            counter += 1
            coins = self.return_coins()
            self.do_move(move)

            if not self.headless:
                pygame.display.set_caption("{:.2f} FPS".format(self.clock.get_fps()))
                self.level.drawLevel(self.mario.camera)
                self.mario.update()
                self.clock.tick(self.max_frame_rate)
                self.dashboard.update()
                pygame.display.update()
                self.score = self.dashboard.points
            else:
                self.level.updateEntities()
                self.mario.update()
                self.clock += (1.0 / 60.0)
                self.score = self.level.points



            #reward = 0.001 * (self.score - start_score + deadbonus)

            reward += coins - self.return_coins()
        # print('The current reward is: {}'.format(reward))
        # print('The extra bonus is: {}'.format(self.mario.rect.x/MAP_MULTIPLIER))

            # if reward > 0:
                # print('He found a coin!!!')
                # print(reward)

        return COIN_REWARD * reward

    def level_to_numpy(self):
        granularity = 8

        padding = int(256/granularity)
        level_size = self.level.levelLength*padding
        array = np.zeros((level_size, level_size))

        array[int(round(self.mario.rect.y/granularity))][int(round(self.mario.rect.x/granularity))] = -1
        # for i, row in enumerate(self.level.level):
        #     for j, ele in enumerate(row):
        #         if ele.rect:
        #             array[i][j] = 5
        for entity in self.level.entityList:
            if entity.__class__.__name__ == 'Koopa' or entity.__class__.__name__ == 'Goomba' or entity.__class__.__name__ == 'GoombaHeadless':
                    array[int(round(entity.rect.y / granularity))][int(round(entity.rect.x / granularity))] = 1
            elif entity.__class__.__name__ == 'Coin':
                array[int(round(entity.rect.y / granularity))][int(round(entity.rect.x / granularity))] = 2
            elif entity.__class__.__name__ == 'RandomBox':
                if not entity.triggered:
                    array[int(round(entity.rect.y / granularity))][int(round(entity.rect.x / granularity))] = 3
                else:
                    array[int(round(entity.rect.y / granularity))][int(round(entity.rect.x / granularity))] = 4
        for ground in self.level.groundList:
            array[int(round(32*ground[1] / granularity))][int(round(32*ground[0] / granularity))] = 5

        array = np.hstack((np.zeros((level_size, padding)), array))
        array = np.vstack((np.zeros((padding, level_size+padding)), array))
        return array[max(0, int(round(self.mario.rect.y / granularity))): max(0,int(round(self.mario.rect.y / granularity))) + 2*padding, int(round(self.mario.rect.x / granularity)):int(round(self.mario.rect.x / granularity)) + 2 * padding]


    def count_entities(self, entity='coin'):

        if self.headless:
            no_entity = len([entity for entity in self.level.entityList if isinstance(entity, CoinHeadless)])
        else:
            no_entity = len([entity for entity in self.level.entityList if isinstance(entity, Coin)])
        self.no_coins = no_entity


    def return_coins(self, entity='coin'):

        if self.headless:
            no_entity = len([entity for entity in self.level.entityList if isinstance(entity, CoinHeadless)])
        else:
            no_entity = len([entity for entity in self.level.entityList if isinstance(entity, Coin)])
        return no_entity

    def level_to_empathic_numpy(self):
        granularity = 8

        padding = int(256/granularity)

        level_size = self.level.levelLength*padding
        array = np.zeros((level_size, level_size))

        mario_pos = [int(round(self.mario.rect.y/granularity)), int(round(self.mario.rect.x/granularity))]

        mario_representation = 128
        ground_representaion = 64
        enemy_representation = 255
        array[mario_pos[0]][mario_pos[1]] = mario_representation

        closest_enemy = None
        closest_distance = np.inf

        for entity in self.level.entityList:
            if entity.__class__.__name__ == 'Koopa' or entity.__class__.__name__ == 'Goomba' or entity.__class__.__name__ == 'GoombaHeadless':
                    enemy_pos = [int(round(entity.rect.y / granularity)), int(round(entity.rect.x / granularity))]
                    array[enemy_pos[0], enemy_pos[1]] = enemy_representation

                    distance = np.linalg.norm(np.array(mario_pos) - np.array(enemy_pos))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_enemy = enemy_pos
            # elif entity.__class__.__name__ == 'Coin':
            #     array[int(round(entity.rect.y / granularity))][int(round(entity.rect.x / granularity))] = 2
            # elif entity.__class__.__name__ == 'RandomBox':
            #     if not entity.triggered:
            #         array[int(round(entity.rect.y / granularity))][int(round(entity.rect.x / granularity))] = 3
            #     else:
            #         array[int(round(entity.rect.y / granularity))][int(round(entity.rect.x / granularity))] = 4
        for ground in self.level.groundList:
            array[int(round(32*ground[1] / granularity))][int(round(32*ground[0] / granularity))] = ground_representaion

        array = np.hstack((np.zeros((level_size, padding)), array))
        array = np.vstack((np.zeros((padding, level_size+padding)), array))

        mario_view = array[max(0, mario_pos[0]): mario_pos[0] + 2*padding, max(0, mario_pos[1]): mario_pos[1] + 2 * padding]
        if closest_enemy:
            enemy_view = array[max(0, closest_enemy[0]): closest_enemy[0] + 2*padding, max(0, closest_enemy[1]): closest_enemy[1] + 2 * padding].copy()
        else:
            enemy_view = mario_view

        enemy_view[enemy_view == mario_representation] = enemy_representation
        enemy_view[padding, padding] = mario_representation
        if mario_view.shape == (padding*2, padding*2) and enemy_view.shape == (padding*2, padding*2):
            total_view = np.dstack((mario_view, enemy_view))
        else:
            total_view = np.zeros((padding*2, padding*2, 2))
        return total_view


    def level_to_supersimple_numpy(self):
        array = np.zeros(6)
        min_distance = 1000000
        for entity in self.level.entityList:
            if entity.__class__.__name__ == 'Goomba' or entity.__class__.__name__ == 'GoombaHeadless':
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
    env = MarioGym(headless=False)
    plt.ion()
    while True:
        if env.mario.restart:
            env.reset()
        else:
            env.do_game_step(np.random.choice(MOVES))
            env.observation = env.level_to_empathic_numpy()

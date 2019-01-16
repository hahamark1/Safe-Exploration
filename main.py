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


MOVES = ['moveLeft', 'moveRight', 'jump', 'jumpLeft', 'jumpRight', 'doNothing']


class Q_Learner():

    def __init__(self, load_q_values=True):
        self.Q_values = {}
        self.levelname = 'Level-basic-with-goombas.json'
        self.load_pickles(load_q_values)
        self.epsilon = 0.1
        self.learning_rate = 0.2
        self.future_discount = 0.99
        self.frame_history = []
        for i in range(1000000):
            self.init_game()
            self.coins_collected.append(0)
            self.goombas_killed.append(0)
            with open('Q_values.pickle1', 'wb') as handle:
                pickle.dump(self.Q_values, handle, protocol=pickle.HIGHEST_PROTOCOL)

            counter = 0
            initial_goombas = len([x for x in self.level.entityList if (x.__class__.__name__ == 'Goomba' and x.alive)])
            initial_coins = len([x for x in self.level.entityList if (x.__class__.__name__ == 'Coin' and x.alive)])

            while (not self.mario.restart):
                self.Q_learning()
                counter += 1
                if counter > 5000:
                    self.mario.restart = True
                if len([x for x in self.level.entityList if (x.__class__.__name__ == 'Goomba' and x.alive)]) == 0:
                    self.mario.restart = True

            self.goombas_killed[-1] = initial_goombas - len(
                [x for x in self.level.entityList if (x.__class__.__name__ == 'Goomba' and x.alive)])
            self.coins_collected[-1] = initial_coins - len(
                [x for x in self.level.entityList if (x.__class__.__name__ == 'Coin' and x.alive)])

            print("Coins collected: {}".format(self.coins_collected))
            print("Goombas killed: {}".format(self.goombas_killed))

            with open('coins_collected.pickle1', 'wb') as handle:
                pickle.dump(self.coins_collected, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('goombas_killed.pickle1', 'wb') as handle:
                pickle.dump(self.goombas_killed, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def init_game(self):
        pygame.mixer.pre_init(44100, -16, 2, 4096)
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        self.max_frame_rate = 600
        self.dashboard = Dashboard("./img/font.png", 8, self.screen)
        self.sound = Sound()
        self.level = Level(self.screen, self.sound, self.dashboard, self.levelname)
        self.menu = Menu(self.screen, self.dashboard, self.level, self.sound)

        self.menu.update()

        self.mario = Mario(0, 0, self.level, self.screen, self.dashboard, self.sound)
        self.clock = pygame.time.Clock()

        self.menu.dashboard.state = "play"
        self.menu.dashboard.time = 0
        self.menu.start = True
        pygame.display.update()

    def load_pickles(self, load_q_values):
        if not load_q_values:
            with open('Q_values.pickle', 'wb') as handle:
                pickle.dump(self.Q_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.coins_collected = []
                self.goombas_killed = []
        else:
            with open('Q_values.pickle1', 'rb') as handle:
                self.Q_values = pickle.load(handle)
            with open('coins_collected.pickle1', 'rb') as handle:
                self.coins_collected = pickle.load(handle)
            with open('goombas_killed.pickle1', 'rb') as handle:
                self.goombas_killed = pickle.load(handle)

    def do_game_step(self, move):
        start_score = self.dashboard.points
        deadbonus = 0
        mario_position = (int(self.mario.rect.y/8), int(self.mario.rect.x/8))
        counter = 0
        while mario_position == (int(self.mario.rect.y/8), int(self.mario.rect.x/8)) and counter < 10:
            counter+=1
            self.do_move(move)
            pygame.display.set_caption("{:.2f} FPS".format(self.clock.get_fps()))
            self.level.drawLevel(self.mario.camera)
            self.mario.update()
            self.clock.tick(self.max_frame_rate)

            pygame.display.update()

            if self.mario.restart:
                deadbonus = -10000
                break
            self.dashboard.update()
            pygame.display.update()


        reward = self.dashboard.points - start_score + deadbonus
        return reward

    def level_to_numpy(self):
        array = np.zeros((600, 600))
        padding = 40
        array[int(round(self.mario.rect.y/8))][int(round(self.mario.rect.x/8))] = -1
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
        array = np.hstack((np.zeros((600, padding-5)), array))
        array = np.hstack((np.ones((600, 5)), array))
        return array[int(round(self.mario.rect.y / 8)) - padding:int(round(self.mario.rect.y / 8)) + padding + 1,
               int(round(self.mario.rect.x / 8)):int(round(self.mario.rect.x / 8)) + 2 * padding]

    def do_move(self, move):
        if move == 'moveLeft':
            self.mario.traits['goTrait'].direction = -1
        elif move == 'moveRight':
            self.mario.traits['goTrait'].direction = 1
        elif move == 'jump':
            self.mario.traits['jumpTrait'].start()
        elif move  == 'jumpRight':
            self.mario.traits['goTrait'].direction = 1
            self.mario.traits['jumpTrait'].start()
        elif move == 'jumpLeft':
            self.mario.traits['goTrait'].direction = -1
            self.mario.traits['jumpTrait'].start()
        elif move == 'doNothing':
            self.mario.traits['goTrait'].direction = 0

    def level_to_key(self):
        level_list = self.level_to_numpy().tolist()
        #print(level_list)
        #level_list.append((int(self.mario.vel.x), np.sign(self.mario.vel.y/10)))
        #if (int(self.mario.vel.x), np.sign(self.mario.vel.y/10)) not in self.frame_history:
        #    self.frame_history.append((int(self.mario.vel.x), np.sign(self.mario.vel.y/10)))
        #    print((int(self.mario.vel.x), np.sign(self.mario.vel.y/10)))
        #    print("length: {}".format(len(self.frame_history)))
        return repr(level_list)

    def get_best_action(self):
        max_Q = -np.inf
        best_action = None
        level_key = self.level_to_key()
        if level_key not in self.Q_values:
            self.Q_values[level_key] = {}
        np.random.shuffle(MOVES)
        for action in MOVES:
            if action not in self.Q_values[level_key]:
                self.Q_values[level_key][action] = 0
            if self.Q_values[level_key][action] >= max_Q:
                max_Q = self.Q_values[level_key][action]
                best_action = action
        return best_action, max_Q

    def Q_learning(self):
        state = self.level_to_key()
        best_action, max_Q = self.get_best_action()
        if np.random.random() > self.epsilon:
            action = best_action
            reward = self.do_game_step(action)
        else:
            action = random.choice(MOVES)
            reward = self.do_game_step(action)

        _, new_Q = self.get_best_action()
        value = reward + self.future_discount * new_Q
        self.Q_values[state][action] = (1-self.learning_rate)*self.Q_values[state][action] + self.learning_rate*value
        #print(self.Q_values[state][action])\
        print(value)



if __name__ == "__main__":
    Q_learner = Q_Learner(load_q_values=True)

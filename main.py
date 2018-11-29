import pygame
import random
import pickle
import numpy as np
from classes.Level import Level
from entities.Mario import Mario
from classes.Dashboard import Dashboard
from classes.Sound import Sound
from classes.Menu import Menu


MOVES = ['moveLeft', 'moveRight', 'jump', 'doNothing']

class Q_Learner():

    def __init__(self):

        self.Q_values = {}
        with open('Q_values.pickle', 'rb') as handle:
            self.Q_values = pickle.load(handle)
        self.init_game()
        self.epsilon = 0.2
        for i in range(1000000):
            #self.epsilon = self.epsilon*0.9
            self.init_game()
            with open('Q_values.pickle', 'wb') as handle:
                pickle.dump(self.Q_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
            counter = 0
            while (not self.mario.restart):
                self.Q_learning()
                counter += 1
                if counter > 10000:
                    self.mario.restart = True

    def init_game(self):
        pygame.mixer.pre_init(44100, -16, 2, 4096)
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        self.max_frame_rate = 60
        self.dashboard = Dashboard("./img/font.png", 8, self.screen)
        self.sound = Sound()
        self.level = Level(self.screen, self.sound, self.dashboard)
        self.menu = Menu(self.screen, self.dashboard, self.level, self.sound)

        self.menu.update()

        self.mario = Mario(0, 0, self.level, self.screen, self.dashboard, self.sound)
        self.clock = pygame.time.Clock()

        self.menu.dashboard.state = "play"
        self.menu.dashboard.time = 0
        self.menu.start = True
        pygame.display.update()


    def do_game_step(self, move):
        start_score = self.dashboard.points
        deadbonus = 0
        mario_position = (int(self.mario.rect.y/32), int(self.mario.rect.x/32))
        counter = 0
        while mario_position == (int(self.mario.rect.y/32), int(self.mario.rect.x/32)) and counter < 50:
            counter+=1
            self.do_move(move)
            pygame.display.set_caption("{:.2f} FPS".format(self.clock.get_fps()))
            self.level.drawLevel(self.mario.camera)
            self.mario.update()
            pygame.display.update()
            self.clock.tick(self.max_frame_rate)
            if self.mario.restart:
                deadbonus = -10000
                break
            self.dashboard.update()
            self.do_move(move)
            self.mario.update()


        reward = self.dashboard.points - start_score + deadbonus -0.00001
        return reward


    def levelToNumpy(self):
        array = np.zeros((60, 60))
        array[int(self.mario.rect.y/32)][int(self.mario.rect.x/32)] = -1
        for i, row in enumerate(self.level.level):
            for j, ele in enumerate(row):
                if ele.rect:
                    array[i][j] = 1
        for entity in self.level.entityList:
            if entity.__class__.__name__ == 'Coin':
                array[entity.getPosIndex().y][entity.getPosIndex().x] = 2
            if entity.__class__.__name__ == 'Koopa' or entity.__class__.__name__ == 'Goomba':
                if entity.getPosIndex().y < 32:
                    array[entity.getPosIndex().y][entity.getPosIndex().x] = 3
            if entity.__class__.__name__ == 'RandomBox':
                if not entity.triggered:
                    array[entity.getPosIndex().y][entity.getPosIndex().x] = 4
                else:
                    array[entity.getPosIndex().y][entity.getPosIndex().x] = 5
        return array[max(0, int(self.mario.rect.y/32)-4):int(self.mario.rect.y/32)+4,
               max(0, int(self.mario.rect.x/32)-3):int(self.mario.rect.x/32)+4]


    def do_move(self, move):
        if move == 'moveLeft':
            self.mario.traits['goTrait'].direction = -1
        elif move == 'moveRight':
            self.mario.traits['goTrait'].direction = 1
        elif move == 'jump':
            self.mario.traits['jumpTrait'].start()
        elif move == 'doNothing':
            self.mario.traits['goTrait'].direction = 0

    def do_random_move(self):
        random_move = random.choice(['moveRight'])
        if random_move == 'moveLeft':
            self.mario.traits['goTrait'].direction = -1
        elif random_move == 'moveRight':
            self.mario.traits['goTrait'].direction = 1
        elif random_move == 'jump':
            self.mario.traits['jumpTrait'].start()
        elif random_move == 'doNothing':
            self.mario.traits['goTrait'].direction = 0

    def level_to_key(self):
        level_list = self.levelToNumpy().tolist()
        level_list.append([self.mario.getPosIndex().x, self.mario.getPosIndex().y])
        level_list.append([np.sign(self.mario.vel.x), np.sign(self.mario.vel.y)])
        return tuple(tuple(x) for x in level_list)

    def get_best_action(self):
        max_Q = -np.inf
        best_action = None
        if self.level_to_key() not in self.Q_values:
            self.Q_values[self.level_to_key()] = {}
        for action in MOVES:
            if action not in self.Q_values[self.level_to_key()]:
                self.Q_values[self.level_to_key()][action] = 20
            if self.Q_values[self.level_to_key()][action] >= max_Q:
                max_Q = self.Q_values[self.level_to_key()][action]
                best_action = action
        return best_action, max_Q

    def Q_learning(self):
        state = self.level_to_key()
        best_action, max_Q = self.get_best_action()
        if np.random.random() > self.epsilon:
            action = best_action
            reward = self.do_game_step(action)
        else:
            action = random.choice(['moveRight'])
            reward = self.do_game_step(action)

        _, new_Q = self.get_best_action()
        value = reward + 0.75 * new_Q
        self.Q_values[state][action] = 0.75*self.Q_values[state][action] + 0.25*value
        #print(self.Q_values[state][action])
        print(value)



if __name__ == "__main__":
    Q_learner = Q_Learner()

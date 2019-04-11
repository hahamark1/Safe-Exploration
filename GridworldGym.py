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

    def __init__(self, headless=True, gridworld_size=11, max_steps=400, kill_reward=0, step_reward=1, dead_reward=0):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-10000000, high=100000000, dtype=np.float, shape=(7, 7, 2))
        plt.ion()
        self.headless = headless
        self.gridworld_size = gridworld_size
        self.max_steps = max_steps
        self.kill_reward = kill_reward
        self.step_reward = step_reward
        self.dead_reward = dead_reward

        self.reset()

    def reset(self):

        self.agent_position = [6, 6]

        self.enemy_positions = [[0,0]]
        self.steps = 0
        self.get_observation()

        return self.observation

    def get_observation(self):
        self.observation = np.zeros((self.gridworld_size, self.gridworld_size))

        self.observation[max(0, min(self.agent_position[0], self.gridworld_size-1)), max(0, min(self.agent_position[1], self.gridworld_size-1))] = 1
        closest_enemy = None
        closest_distance = np.inf
        for pos in self.enemy_positions:
            self.observation[max(0, min(pos[0], self.gridworld_size-1)), max(0, min(pos[1], self.gridworld_size-1))] = -1
            distance = np.linalg.norm(np.array(pos)-np.array(self.agent_position))
            if distance < closest_distance:
                closest_distance = distance
                closest_enemy = pos

        window = [range(self.agent_position[0] - 3, self.agent_position[0] + 4), range(self.agent_position[1] - 3, self.agent_position[1] + 4)]
        agent_observation = self.observation.take(window[0], axis=0, mode='wrap')

        agent_observation = agent_observation.take(window[1], axis=1, mode='wrap')

        if closest_enemy:
            enemy_window = [range(closest_enemy[0] - 3, closest_enemy[0] + 4), range(closest_enemy[1] - 3, closest_enemy[1] + 4)]
            enemy_observation = self.observation.take(enemy_window[0], axis=0, mode='wrap')
            enemy_observation = enemy_observation.take(enemy_window[1], axis=1, mode='wrap')

            # Swap enemies and agents
            enemy_observation = enemy_observation*-1

            #for now I also remove all the other enemies, TODO: test without this
            enemy_observation = np.clip(enemy_observation, -1, 0)
            enemy_observation[3,3] = 1
        else:
            enemy_observation = np.zeros((7,7))

        self.observation = np.dstack([agent_observation, enemy_observation])
        return self.observation

    def plot_env(self):
        self.observation = self.get_observation()

        env = np.zeros((self.gridworld_size, self.gridworld_size))

        env[max(0, min(self.agent_position[0], self.gridworld_size-1)), max(0, min(self.agent_position[1], self.gridworld_size-1))] = 1
        for pos in self.enemy_positions:
            env[max(0, min(pos[0], self.gridworld_size-1)), max(0, min(pos[1], self.gridworld_size-1))] = -1
        if not self.headless:
            plt.matshow(self.observation[:,:,0], 1)
            plt.draw()
            plt.pause(0.1)
            plt.matshow(self.observation[:,:,1], 2)
            plt.draw()
            plt.pause(0.1)
            plt.clf()

    def step(self, action_num):

        #increase counter
        self.steps += 1

        #update agents
        dead = (self.agent_position in [[x[0] +1, x[1]] for x in self.enemy_positions]) or (self.agent_position in [[x[0], x[1] +1] for x in self.enemy_positions])
        num_enemies_killed = self.kill_enemies()

        if not dead and self.steps % 2 == 0:

            #move agent
            if action_num == 0:
                self.agent_position = [(self.agent_position[0] + 1) % self.gridworld_size, self.agent_position[1]]
            elif action_num == 1:
                self.agent_position = [(self.agent_position[0] - 1) % self.gridworld_size, self.agent_position[1]]
            elif action_num == 2:
                self.agent_position = [self.agent_position[0], (self.agent_position[1] + 1) % self.gridworld_size]
            elif action_num == 3:
                self.agent_position = [self.agent_position[0], (self.agent_position[1] - 1) % self.gridworld_size]

            if not self.headless:
                self.plot_env()

        elif not dead:
            #move enemies
            for i, pos in enumerate(self.enemy_positions):
                action = np.random.choice(range(4))
                if action == 0:
                    pos = [(pos[0] + 1) % self.gridworld_size, pos[1]]
                elif action == 1:
                    pos = [(pos[0] - 1) % self.gridworld_size, pos[1]]
                elif action == 2:
                    pos = [pos[0], (pos[1] + 1) % self.gridworld_size]
                elif action == 3:
                    pos = [pos[0], (pos[1] - 1) % self.gridworld_size]
                self.enemy_positions[i] = pos

            if not self.headless:
                self.plot_env()
                a=1

        if dead:
            self.reset()
            if not self.headless:
                plt.pause(3.0)

        reward = self.step_reward + self.kill_reward*num_enemies_killed + self.dead_reward*dead
        info = {'num_killed': num_enemies_killed, 'got_killed': dead}

        restart = dead or self.steps > self.max_steps

        self.observation = self.get_observation()

        return self.observation, reward, restart, info

    def render(self, mode='human', close=False):
        pass

    def kill_enemies(self):
        enemies_to_be_killed = []
        for enemy_pos in self.enemy_positions:
            if enemy_pos == [self.agent_position[0] +1, self.agent_position[1]] or enemy_pos == [self.agent_position[0], self.agent_position[1] + 1]:
                enemies_to_be_killed.append(enemy_pos)
        self.enemy_positions = [x for x in self.enemy_positions if x not in enemies_to_be_killed]
        return len(enemies_to_be_killed)


if __name__ == "__main__":
    env = GridworldGym()


    while True:
        action = np.random.choice(range(4))
        env.step(action)

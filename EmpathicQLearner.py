import pygame
import random
import pickle
import numpy as np
import tensorflow as tf

from multiprocessing import Process

from datetime import datetime

from GridworldGym import GridworldGym

env = GridworldGym()

class EmphaticQLearner():

    def __init__(self, load_q_values=True):
        self.Q_values = {}
        self.load_pickles(load_q_values)
        self.epsilon = 0.99
        self.learning_rate = 0.05
        self.future_discount = 0.99
        self.selfishness = 0.5
        self.writer = tf.summary.FileWriter(f'logs/LRLearning3.0/{str(datetime.now())}')
        self.step = 0
        self.log_q_values=[[]]

        for i in range(10000000):
            self.step += 1
            if self.epsilon > 0.1:
                self.epsilon = self.epsilon * 0.999999

            self.Q_learning()

            EmpathicQLearner.py


    def log_scalar(self, tag, value, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_histogram(self, tag, values, global_step, bins):
        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary()
        summary.value.add(tag=tag, histo=hist)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def load_pickles(self, load_q_values):
        if not load_q_values:
            with open('Q_values.pickle', 'wb') as handle:
                pickle.dump(self.Q_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.rewards = [0]
                self.enemies_killed = [0]
        else:
            with open('Q_values.pickle', 'rb') as handle:
                self.Q_values = pickle.load(handle)
            with open('coins_collected.pickle', 'rb') as handle:
                self.rewards = pickle.load(handle)
            with open('goombas_killed.pickle', 'rb') as handle:
                self.enemies_killed = pickle.load(handle)

    def do_game_step(self, move):

        next_total_state, reward, done, info = env.step(move)
        if done:
            env.reset()
            self.log_scalar('reward', self.rewards[-1], self.step)
            self.log_scalar('kills', self.enemies_killed[-1], self.step)
            self.log_scalar('epsilon', self.epsilon, self.step)
            self.log_scalar('mean_q', np.mean(self.log_q_values[-1]), self.step)
            self.log_histogram('q_values', np.array(self.log_q_values[-1]), self.step, 20)
            self.rewards.append(0)
            self.enemies_killed.append(0)
            self.log_q_values.append([])

        return reward, done, info

    def level_to_key(self):
        obs = env.get_observation()
        obs1 = tuple(map(tuple, obs[:, :, 0]))
        obs2 = tuple(map(tuple, obs[:, :, 1]))
        return obs1, obs2


    def get_best_action(self):
        max_Q = -np.inf
        best_action = None
        level_key, enemy_level_key = self.level_to_key()
        if level_key not in self.Q_values:
            self.Q_values[level_key] = {}
        if enemy_level_key not in self.Q_values:
            self.Q_values[enemy_level_key] = {}
        for action in range(4):
            if action not in self.Q_values[level_key]:
                self.Q_values[level_key][action] = 100
            if action not in self.Q_values[enemy_level_key]:
                self.Q_values[enemy_level_key][action] = 100
            if self.Q_values[level_key][action] >= max_Q:
                max_Q = self.Q_values[level_key][action]
                best_action = action
        enemy_V = np.mean(list(self.Q_values[enemy_level_key].values()))
        return best_action, max_Q, enemy_V

    def Q_learning(self):
        state, enemy_state = self.level_to_key()
        best_action, max_Q, enemy_V = self.get_best_action()
        if np.random.random() > self.epsilon:
            action = best_action
            reward, done, info = self.do_game_step(action)
        else:
            action = random.choice(range(4))
            reward, done, info = self.do_game_step(action)

        _, new_Q, new_enemy_V = self.get_best_action()
        value = self.selfishness * (1 - done) * (reward + self.future_discount * new_Q) + (1 - self.selfishness) * new_enemy_V
        self.Q_values[state][action] = (1-self.learning_rate)*self.Q_values[state][action] + self.learning_rate*value

        self.rewards[-1] += reward
        self.enemies_killed[-1] += info['num_killed']
        self.log_q_values[-1].append(max_Q)
        print(f'average reward: {np.mean(self.rewards)},    epsilon: {self.epsilon}')



if __name__ == "__main__":
    Q_learner = EmphaticQLearner(load_q_values=False)

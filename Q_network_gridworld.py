import pygame
import random
import pickle
import numpy as np
import tensorflow as tf

from multiprocessing import Process

from datetime import datetime

from GridworldGym import GridworldGym

env = GridworldGym(headless=True, dynamic_holes=True)

class EmphaticQLearner():

    def __init__(self, load_q_values=True):
        self.Q_values = {}
        self.load_pickles(load_q_values)
        self.epsilon = 0.99
        self.learning_rate = 0.05
        self.future_discount = 0.99
        self.selfishness = 0.5
        self.writer = tf.summary.FileWriter(f'logs/Q_Tab_Grid/{str(datetime.now())}')
        self.step = 0
        self.episodes = 0
        self.log_q_values=[[]]
        self.total_death = 0
        self.total_succeed = 0
        for i in range(10000000):
            self.step += 1
            if self.epsilon > 0.1:
                self.epsilon = self.epsilon * 0.999999

            self.Q_learning()


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
        else:
            with open('Q_values.pickle', 'rb') as handle:
                self.Q_values = pickle.load(handle)
            with open('coins_collected.pickle', 'rb') as handle:
                self.rewards = pickle.load(handle)
            with open('goombas_killed.pickle', 'rb') as handle:
                self.enemies_killed = pickle.load(handle)

    def do_game_step(self, move):

        next_state, reward, done, info = env.step(move)

        if done:
            env.reset()
            self.rewards[-1] += reward
            self.log_scalar('reward', self.rewards[-1], self.step)
            self.log_scalar('epsilon', self.epsilon, self.step)
            self.log_scalar('mean_q', np.mean(self.log_q_values[-1]), self.step)
            self.log_histogram('q_values', np.array(self.log_q_values[-1]), self.step, 20)
            self.rewards.append(0)
            self.log_q_values.append([])

        return next_state, reward, done, info

    def level_to_key(self, obs):
        obs1 = tuple(map(tuple, obs))
        return obs1


    def get_best_action(self, state):
        max_Q = -np.inf
        best_action = None
        if state not in self.Q_values:
            self.Q_values[state] = {}
        for action in range(4):
            if action not in self.Q_values[state]:
                self.Q_values[state][action] = 1
            if self.Q_values[state][action] >= max_Q:
                max_Q = self.Q_values[state][action]
                best_action = action
        return best_action, max_Q

    def Q_learning(self):
        state = env.get_observation()
        state_key = self.level_to_key(state)
        best_action, max_Q = self.get_best_action(state_key)
        if np.random.random() > self.epsilon:
            action = best_action
            next_state, reward, done, info = self.do_game_step(action)
        else:
            action = random.choice(range(4))
            next_state, reward, done, info = self.do_game_step(action)

        new_state_key = self.level_to_key(next_state)
        _, new_Q = self.get_best_action(new_state_key)
        value =  (reward + self.future_discount * new_Q)
        self.Q_values[state_key][action] = (1-self.learning_rate)*self.Q_values[state_key][action] + self.learning_rate*value

        # self.rewards[-1] += reward
        if 'death' in info:
            self.total_death += info['death']
        if 'succeed' in info:
            self.total_succeed += info['succeed']
        self.log_q_values[-1].append(max_Q)
        if done:
            self.episodes += 1
            if len(self.rewards) > 500:
                average_last = np.mean(self.rewards[-500:])
            else:
                average_last = np.mean(self.rewards)
            print(f'Currently at episode: {self.episodes},     average rewardlast 500: {average_last},    last reward: {self.rewards[-2]},    epsilon: {self.epsilon},     total death: {self.total_death},       total succeed: {self.total_succeed}')



if __name__ == "__main__":
    Q_learner = EmphaticQLearner(load_q_values=False)

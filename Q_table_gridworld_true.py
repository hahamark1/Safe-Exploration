import pygame
import random
import pickle
import numpy as np
import tensorflow as tf
import _datetime
from joblib import Parallel, delayed

from multiprocessing import Process

from datetime import datetime

from GridworldGym import GridworldGym



def smooth(x, N):
    x = np.array(x)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

class EmphaticQLearner():

    def __init__(self, load_q_values=True, gridworld_size=7):
        self.Q_values = {}
        self.load_pickles(load_q_values)
        self.epsilon = 0.99
        self.learning_rate = 0.05
        self.gridworld_size = gridworld_size
        self.future_discount = 0.99
        self.selfishness = 0.5
        self.writer = tf.summary.FileWriter(f'logs/Q_Tab_Grid/{str(datetime.now())}')
        self.step = 0
        self.env = GridworldGym(headless=True, dynamic_holes=True, dynamic_start=False, constant_change=False, gridworld_size=gridworld_size)
        self.episodes = 0
        self.episode_setps = 0
        self.episode_durations = []
        self.log_q_values=[[]]
        # self.rewards = []
        self.total_death = 0
        self.total_succeed = 0


    def train(self):
        while self.episodes < 10000:
            self.step += 1
            if self.epsilon > 0.1:
                self.epsilon = self.epsilon * 0.9999

            self.Q_learning()


    def save_rewards(self):

        fn = 'big_chart_pickles/{}_{}_{}.pt'.format(self.gridworld_size, 'Table',
                                                 datetime.now().timestamp())

        with open(fn, "wb") as pf:
            pickle.dump((self.rewards, self.gridworld_size, 'Table', self.episode_durations), pf)
        print('Saved an experiment for the Q_table with size {}.'.format(self.gridworld_size))


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

        next_state, reward, done, info = self.env.step(move)

        if done:
            self.env.reset()
            self.rewards[-1] += reward
            self.log_scalar('reward', self.rewards[-1], self.step)
            self.log_scalar('epsilon', self.epsilon, self.step)
            self.log_scalar('mean_q', np.mean(self.log_q_values[-1]), self.step)
            # self.log_histogram('q_values', np.array(self.log_q_values[-1]), self.step, 20)
            self.rewards.append(0)
            self.log_q_values.append([])

        return next_state, reward, done, info

    def level_to_key(self, obs):
        obs1 = tuple(map(tuple, obs))
        return obs1


    def get_best_action(self, state):
        max_Q = -np.inf
        best_action = None
        state_x, state_y = state[0], state[1]
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

        state = tuple(self.env.agent_position)
        best_action, max_Q = self.get_best_action(state)

        if np.random.random() > self.epsilon:
            action = best_action
            next_state, reward, done, info = self.do_game_step(action)
        else:
            action = random.choice(range(4))
            next_state, reward, done, info = self.do_game_step(action)

        next_state = tuple(self.env.agent_position)

        _, new_Q = self.get_best_action(next_state)
        value =  (reward + self.future_discount * new_Q)
        self.Q_values[state][action] = (1-self.learning_rate)*self.Q_values[state][action] + self.learning_rate*value
        self.episode_setps += 1
        # self.rewards[-1] += reward
        if 'death' in info:
            self.total_death += info['death']
        if 'succeed' in info:
            self.total_succeed += info['succeed']
        self.log_q_values[-1].append(max_Q)
        if done:
            self.episodes += 1

            self.episode_durations.append(self.episode_setps)
            self.episode_setps = 0
            if len(self.rewards) > 500:
                average_last = np.mean(self.rewards[-500:])
            else:
                average_last = np.mean(self.rewards)
            # print(f'Currently at episode: {self.episodes},     average rewardlast 500: {average_last},    last reward: {self.rewards[-2]},    epsilon: {self.epsilon},     total death: {self.total_death},       total succeed: {self.total_succeed}')

def run_Q_learner(gridworld_size):
    Trainer = EmphaticQLearner(load_q_values=False, gridworld_size=gridworld_size)
    Trainer.train()
    # print(np.max(smooth(Trainer.rewards, 100)))
    Trainer.save_rewards()


if __name__ == "__main__":
    # run_Q_learner(6)
    gridworld_sizes = [x for x in range(3, 33)]
    # # embeddings = [True, False]
    number_of_experiments = 10
    #
    Parallel(n_jobs=4)(
        delayed(run_Q_learner)(size) for size in gridworld_sizes for i in
        range(number_of_experiments))
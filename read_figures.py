import os

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import datetime

from Q_network_gridworld import smooth


def plot_results(key, data, smooth_factor=100, fig_folder='figures/'):
    fig = plt.figure()
    plt.plot(smooth(data['Episode_durations'], smooth_factor))
    plt.title('Episode durations per episode')
    fig.savefig('{}/{}_ep_dur.png'.format(fig_folder, key))

    fig = plt.figure()
    plt.plot(smooth(data['Rewards'], smooth_factor))
    plt.title('Reward per episode')
    fig.savefig('{}/{}_rewards.png'.format(fig_folder, key))

    fig = plt.figure()
    plt.plot(data['Number of Deaths'])
    plt.title('Number of deaths over time')
    fig.savefig('{}/{}_nem_deaths.png'.format(fig_folder, key))

    # print('The current number of deaths is {} after {} episodes'.format(self.num_deaths, self.episode_number))
    plt.close()

figures_path = '/home/hahamark/Desktop'
data = {}

for file in os.listdir(figures_path):
    if file.endswith('.pt'):
        with open('{}/{}'.format(figures_path, file), 'rb') as pf:
            data[file] = pickle.load(pf)

for key in data.keys():
    plot_results(key, data[key])

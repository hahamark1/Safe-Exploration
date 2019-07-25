import pickle
import os
import numpy as np

import matplotlib.pyplot as plt
chart_pickles = 'big_chart_pickles'


keys = ['Reward', 'Gridworld_size', 'Network_type', 'Episode_durations']


def load_data(folder, smoothing_factor=100):
    data = {}
    for file in os.listdir(folder):
        name_parts = file.split('_')
        network = name_parts[1]
        gw_size = name_parts[0]
        with open('{}/{}'.format(folder, file), 'rb') as pf:
            content = pickle.load(pf)
        if network not in data:
            data[network] = {}
        if len(content) == len(keys):
            if gw_size not in data[network]:
                data[network][gw_size] = {}
                for i in range(len(content)):
                    data[network][gw_size][keys[i]] = []
            for i in range(len(content)):
                if isinstance(content[i], list):
                    data[network][gw_size][keys[i]].append(smooth(content[i], smoothing_factor))
    return data

def smooth(x, N):
    x = np.array(x)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def prepare_data(data):

    prepared_data = {}

    learn_curve = {}
    for network in data.keys():
        if network not in prepared_data:
            prepared_data[network] = {}
            learn_curve[network] = {}

        for gw_size in data[network].keys():
            if gw_size not in prepared_data[network]:
                prepared_data[network][gw_size] = {}
                learn_curve[network][gw_size] = {}

            rewards = np.array(data[network][gw_size]['Reward'])
            episode_durations = np.array(data[network][gw_size]['Episode_durations'])

            print(rewards.shape)
            print(network, gw_size)
            mean_rewards = np.mean(rewards, axis=0)


            mean_episode_durations = np.mean(episode_durations, axis=0)

            prepared_data[network][gw_size]['Reward'] = mean_rewards
            print(np.array(mean_rewards))

            er
            prepared_data[network][gw_size]['Episode_durations'] = mean_episode_durations
            learn_curve[network][gw_size] = np.argmax(mean_rewards>0.8)

    for network in learn_curve.keys():
        print([x for x in sorted([int(x) for x in learn_curve[network].keys()])])
        learn_curve[network] = [learn_curve[network][str(gw_size)] for gw_size in sorted([int(x) for x in learn_curve[network].keys()])]

    return prepared_data, learn_curve



if __name__ == '__main__':
    data = load_data(chart_pickles)
    prepared_data, learn_curve = prepare_data(data)
    print(learn_curve)

    for network in learn_curve.keys():
        plt.plot(learn_curve[network])
    plt.show()


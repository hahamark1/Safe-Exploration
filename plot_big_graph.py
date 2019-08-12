import pickle
import os
import numpy as np

import matplotlib.pyplot as plt
chart_pickles = 'big_chart_pickles'
import matplotlib
from distutils import util

colors = [name for name, hex in matplotlib.colors.cnames.items()]

color_1 = 'cornflowerblue'
color_2 = 'orange'
color_3 = 'lime'
color_4 = 'violet'
colors = [color_1, color_2, color_3, color_4]

keys = ['Reward', 'Gridworld_size', 'Network_type', 'Episode_durations', 'Number of Kills']

namen = {'Table': 'Q-learning, simple representation', 'Table_full_state': 'Q-learning, full state representation', 'SimpleCNN':'Deep Q-learning, small convolutional network',
            'QNetwork': 'Deep Q-learning, fully connected network', 'DQN':'Deep Q-learning, large convolutional network', 'type': 'IDK'}

# namen = {'Table': 'QL', 'Table_full_state': 'Full QL', 'SimpleCNN':'DQN CNN',
            # 'QNetwork': 'DQN FC', 'DQN':'DQN CNN B', 'type': 'IDK'}

networks = ['Table', 'Table_full_state', 'QNetwork', 'SimpleCNN']

dyn_networks = ['QNetwork', 'SimpleCNN']
def load_data(folder, smoothing_factor=100):
    data = {}
    for file in os.listdir(folder):
        name_parts = file.split('_')
        network = ('_').join(name_parts[1:-1])
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
                    if len(content[i] == 10000):
                        data[network][gw_size][keys[i]].append(content[i])
    return data

def load_sup_data(folder, smoothing_factor=100):
    data = {}
    for file in os.listdir(folder):
        name_parts = file.split('_')
        network = name_parts[1]
        if network in dyn_networks:
            gw_size = name_parts[0]
            sup = name_parts[2]
            mem = name_parts[3]
            dh = name_parts[4]
            with open('{}/{}'.format(folder, file), 'rb') as pf:
                content = pickle.load(pf)
            if len(content) == len(keys):
                if network not in data:
                    data[network] = {}
                # if len(content) == len(keys):
                if gw_size not in data[network]:
                    data[network][gw_size] = {}
                if dh not in data[network][gw_size]:
                    data[network][gw_size][dh] = {}
                if '{}{}'.format(sup, mem) not in data[network][gw_size][dh]:
                    data[network][gw_size][dh]['{}{}'.format(sup, mem)] = {}
                    for i in range(len(content)):
                        data[network][gw_size][dh]['{}{}'.format(sup,mem)][keys[i]] = []
                for i in range(len(content)):
                    if isinstance(content[i], list):
                        if len(content[i]) == 10000:
                            data[network][gw_size][dh]['{}{}'.format(sup,mem)][keys[i]].append(content[i])
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

            # print(rewards.shape)
            # print(network, gw_size)
            mean_rewards = np.mean(rewards, axis=0)
            print('currently workin gon {}'.format(network))
            print(rewards   )
            plt.plot(mean_rewards)
            plt.show()


            mean_episode_durations = np.mean(episode_durations, axis=0)
            num_kills = np.array([np.unique(rew, axis=0, return_counts=True)[1][np.where(np.unique(rew, axis=0, return_counts=True)[0] == -1)] for rew in rewards])
            # print(np.array([np.unique(rew, axis=0, return_counts=True)[1] for rew in rewards]))
            # print(np.array([np.unique(rew, axis=0, return_counts=True)[0] for rew in rewards]))
            prepared_data[network][gw_size]['Reward'] = mean_rewards
            num_succeed = np.array([np.unique(rew, axis=0, return_counts=True)[1][np.where(np.unique(rew, axis=0, return_counts=True)[0] == 1)] for rew in rewards])
            print('For gw {} and network {} the total number of wins is {}'.format(gw_size, network, np.mean(num_succeed[0])))

            prepared_data[network][gw_size]['Episode_durations'] = mean_episode_durations
            prepared_data[network][gw_size]['Mean_kills'] = np.mean(num_kills)
            prepared_data[network][gw_size]['Std_kills'] = np.std(num_kills)
            learn_curve[network][gw_size] = np.argmax(mean_rewards>0.8)
    # print(prepared_data.keys())
    for network in prepared_data.keys():
        # print([x for x in sorted([int(x) for x in learn_curve[network].keys()])])
        # learn_curve[network] = [learn_curve[network][str(gw_size)] for gw_size in sorted([int(x) for x in learn_curve[network].keys()])]
        prepared_data[network]['Mean_kills'] = np.array([prepared_data[network][str(gw_size)]['Mean_kills'] for gw_size in sorted([int(x) for x in prepared_data[network].keys() if x.isdigit()])])
        prepared_data[network]['Std_kills'] = np.array([prepared_data[network][str(gw_size)]['Std_kills'] for gw_size in
                                                sorted([int(x) for x in prepared_data[network].keys() if x.isdigit()])])
    return prepared_data, learn_curve

def prepare_sup_data(data):

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

            for dh in data[network][gw_size].keys():
                if dh not in prepared_data[network][gw_size]:
                    prepared_data[network][gw_size][dh] = {}

                for mem in data[network][gw_size][dh].keys():
                    if mem not in prepared_data[network][gw_size][dh]:
                        prepared_data[network][gw_size][dh][mem] = {}

                    rewards = np.array(data[network][gw_size][dh][mem]['Reward'])
                    if rewards.any():
                        episode_durations = np.array(data[network][gw_size][dh][mem]['Episode_durations'])

                        # print(rewards.shape)
                        # print(network, gw_size)
                        fig = plt.figure()
                        mean_rewards = np.mean(rewards, axis=0)
                        # print('currently workin gon {}'.format(network))
                        # # print(rewards)
                        # fig.suptitle('For gw {}, dh = {}, mem = {} and network {}'.format(gw_size,dh, mem, network), fontsize=10)
                        # plt.plot(smooth(mean_rewards, 100))
                        #
                        # plt.show()


                        mean_episode_durations = np.mean(episode_durations, axis=0)
                        num_kills = np.array([np.unique(rew, axis=0, return_counts=True)[1][np.where(np.unique(rew, axis=0, return_counts=True)[0] == -1)] for rew in rewards])
                        # print(np.array([np.unique(rew, axis=0, return_counts=True)[1] for rew in rewards]))
                        # print(np.array([np.unique(rew, axis=0, return_counts=True)[0] for rew in rewards]))
                        prepared_data[network][gw_size][dh][mem]['Reward'] = mean_rewards
                        num_succeed = np.array([np.unique(rew, axis=0, return_counts=True)[1][np.where(np.unique(rew, axis=0, return_counts=True)[0] == 1)] for rew in rewards])
                        print('For gw {}, dh = {}, mem = {} and network {} the total number of wins is {}'.format(gw_size,dh, mem, network, np.mean(num_succeed)))

                        prepared_data[network][gw_size][dh][mem]['Episode_durations'] = mean_episode_durations
                        prepared_data[network][gw_size][dh][mem]['Mean_kills'] = np.mean(num_kills)
                        prepared_data[network][gw_size][dh][mem]['Std_kills'] = np.std(num_kills)
                        # learn_curve[network][gw_size] = np.argmax(mean_rewards>0.8)
    # print(prepared_data.keys())
    for network in prepared_data.keys():
        for dh in prepared_data[network]['15'].keys():
            for mem in prepared_data[network]['15'][dh].keys():
                # print([x for x in sorted([int(x) for x in learn_curve[network].keys()])])
                if mem != 'sup=Truelm=True':
                    if dh not in prepared_data[network]:
                        prepared_data[network][dh] = {}
                    if mem not in prepared_data[network][dh]:
                        prepared_data[network][dh][mem] = {}
                    # learn_curve[network] = [learn_curve[network][str(gw_size)] for gw_size in sorted([int(x) for x in learn_curve[network].keys()])]
                    print(network, mem, dh)
                    prepared_data[network][dh][mem]['Mean_kills'] = np.array([prepared_data[network][str(gw_size)][dh][mem]['Mean_kills'] for gw_size in sorted([int(x) for x in prepared_data[network].keys() if x.isdigit() if mem in prepared_data[network][x][dh]])])
                    prepared_data[network][dh][mem]['Std_kills'] = np.array([prepared_data[network][str(gw_size)][dh][mem]['Std_kills'] for gw_size in
                                                            sorted([int(x) for x in prepared_data[network].keys() if x.isdigit() if mem in prepared_data[network][x][dh]])])
    return prepared_data, learn_curve

def debug_data(data, prepared_data):
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

def plot_data(prepared_data, i=0):
    fig = plt.figure()
    xes = [3,7,15,24]
    ax = fig.add_subplot(111)

    for network in prepared_data.keys():
        for dh in prepared_data[network]['15'].keys():
            for mem in prepared_data[network][dh].keys():
                if mem == 'sup=Truelm=False':
                    std = 0.5 * prepared_data[network][dh][mem]['Std_kills']
                    upper_bound =  prepared_data[network][dh][mem]['Mean_kills'] + std
                    lower_bound = prepared_data[network][dh][mem]['Mean_kills'] - std
                    x = xes[:len(std)]
                    plt.plot(x, prepared_data[network][dh][mem]['Mean_kills'], 'k', c=colors[i], label='{} in a {} gridworld'.format(namen[network], 'dynamic' if util.strtobool(dh.split('=')[1]) else 'static'))
                    # print(namen[network])
                    plt.fill_between(x, lower_bound, upper_bound, alpha=0.4, edgecolor=colors[i], facecolor=colors[i])
                    i+=1

    ax.set_xlabel('Gridworld size')
    ax.set_ylabel('Number of kills')
    ax.set_title('The number of kills over 10000 episodes per gridworld size \nusing Supervised Demonstration Learning.')
    ax.legend()
    # ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.subplots_adjust(right=0.9)
    plt.grid()
    # plt.show()
    plt.savefig('figures/Dem_chart.png')
    i=0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for network in prepared_data.keys():
        for dh in prepared_data[network]['15'].keys():
            for mem in prepared_data[network][dh].keys():
                if mem == 'sup=Falselm=True':
                    std = 0.5 * prepared_data[network][dh][mem]['Std_kills']
                    upper_bound =  prepared_data[network][dh][mem]['Mean_kills'] + std
                    lower_bound = prepared_data[network][dh][mem]['Mean_kills'] - std
                    x = xes[:len(std)]
                    plt.plot(x, prepared_data[network][dh][mem]['Mean_kills'], 'k', c=colors[i], label='{} in a {} gridworld'.format(namen[network], 'dynamic' if util.strtobool(dh.split('=')[1]) else 'static'))
                    # print(namen[network])
                    plt.fill_between(x, lower_bound, upper_bound, alpha=0.4, edgecolor=colors[i], facecolor=colors[i])
                    i+=1

    ax.set_xlabel('Gridworld size')
    ax.set_ylabel('Number of kills')
    ax.set_title('The number of kills over 10000 episodes per gridworld size \nusing Demonstration Learning by Memory.')
    ax.legend()
    # ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.subplots_adjust(right=0.9)
    plt.grid()
    # plt.show()

    plt.savefig('figures/Mem_chart.png')

if __name__ == '__main__':
    # data = load_data(chart_pickles)
    # prepared_data, learn_curve = prepare_data(data)
    # plot_data(prepared_data)
    # debug_data(data, prepared_data)

    # for network in ['Table_full_state', 'Table']:
    #     plt.plot(learn_curve[network])
    # plt.show()

    data = load_sup_data('supervised_pickles/')
    prepared_data, learn_curve = prepare_sup_data(data)
    plot_data(prepared_data)

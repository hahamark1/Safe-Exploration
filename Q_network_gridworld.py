import os

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import datetime

from tqdm import tqdm as _tqdm
from GridworldGym import GridworldGym

import random
from constants_gridwolrd import *
from models import *
NETWORK = QNetwork
from joblib import Parallel, delayed

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer


assert sys.version_info[:3] >= (3, 6, 0), "Make sure you have Python 3.6 installed!"

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        # YOUR CODE HERE
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)

    def sample(self, batch_size):
        # YOUR CODE HERE
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




def get_epsilon(it):
    if it < 1000:
        return (1 - 0.95*(it/1000))
    else:
        return 0.05


def select_action(model, state, epsilon):
    with torch.no_grad():
        state = np.expand_dims(state, axis=0)
        state_tensor = torch.tensor(state).type(torch.FloatTensor).to(device)
        out = model(state_tensor)
        if random.uniform(0, 1) <= epsilon:
            return random.choice([0, 1])
        else:
            _, index = out.max(-1)
            index = int(index.item())

    return index


def compute_q_val(model, state, action):
    # YOUR CODE HERE
    q_value = model(state)
    targets = q_value[np.arange(len(q_value)), action]
    return targets


def compute_target(model, reward, next_state, done, discount_factor):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
    # YOUR CODE HERE

    next_state_tensor = next_state

    done = 1 - done

    max_value, _ = model(next_state_tensor).max(1)
    Q_target = reward + done.float() * discount_factor * max_value

    return Q_target

def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def check_conv_net(net):
    conv_net = False
    for layer in net.children():
        if isinstance(layer, nn.Conv2d):
            conv_net = True
    return conv_net

class trainer_Q_network(object):
    def __init__(self, network=NETWORK, num_episodes=NUM_EPISODES,
                 memory_size=MEMORY_SIZE, seed=SEED, discount_factor=DISCOUNT_FACTOR,
                 headless=True, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                 dynamic_holes=DYNAMIC_HOLES, dynamic_start=DYNAMIC_START, load_episode=False, save_every=SAVE_EVERY,
                 plot_every=PLOT_EVERY, plotting=False, embedding=EMBEDDING, gridworld_size=7):

        self.num_episodes = num_episodes
        self.memory = ReplayMemory(memory_size)
        self.discount_factor = discount_factor


        if embedding and network.__class__.__name__ != 'SimpleCNN':
            self.env = GridworldGym(headless=headless, dynamic_holes=dynamic_holes, dynamic_start=dynamic_start, embedding=embedding)
            self.network = network(embedding=embedding).to(device)
        else :
            self.env = GridworldGym(headless=headless, dynamic_holes=dynamic_holes, dynamic_start=dynamic_start)
            self.network = network().to(device)
        self.initialize(seed)
        self.batch_size = batch_size
        self.save_every = save_every
        self.plot_every = plot_every
        self.embedding = embedding
        self.optimizer = optim.Adam(self.network.parameters(), learning_rate)
        self.episode_number = 0
        self.num_deaths = 0
        self.episode_durations = []
        self.number_of_deaths = []
        self.rewards = []
        self.steps = 0
        self.loss = 0
        self.experiment_name = 'checkpoint_{}_DH={}_DS={}_em={}'.format(self.network.__class__.__name__, dynamic_holes, dynamic_start, self.embedding)
        self.exp_folder = 'checkpoints'
        self.fig_folder = 'figures'
        self.smooth_factor = 100

        if self.embedding and not check_conv_net(self.network):
            raise ValueError('We cannot combine embeddings with a Feed-Forward network!')

        if load_episode:
            self.load_model(load_episode)
        # if not plotting:
        #     self.run_episodes()
        # else:
        #     self.plot_results()

    def initialize(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        # self.env.seed(seed)

    def train(self):
        # DO NOT MODIFY THIS FUNCTION

        # don't learn without some decent experience
        if len(self.memory) < self.batch_size:
            return None

        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.batch_size)

        # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
        state, action, reward, next_state, done = zip(*transitions)

        # convert to PyTorch and define types
        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.int64).to(device)  # Need 64 bit to use them as index

        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.uint8).to(device)  # Boolean

        # compute the q value
        q_val = compute_q_val(self.network, state, action)

        with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
            target = compute_target(self.network, reward, next_state, done, self.discount_factor)

        # loss is measured from error between current and newly expected Q values
        loss = F.smooth_l1_loss(q_val, target)

        # backpropagation of loss to Neural Network (PyTorch magic)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_episodes(self):
         # Count the steps (do not reset at episode start, to compute epsilon)

        while self.episode_number < self.num_episodes:
            if self.episode_number % PRINT_EVERY == 0:
                print('Currently working on episode {}'.format(self.episode_number))
            done = False
            episode_duration = 0
            self.episode_number += 1
            s = self.env.reset()
            rew = 0

            while not done:
                epsilon = get_epsilon(self.steps)
                episode_duration += 1
                self.steps += 1
                a = select_action(self.network, s, epsilon)
                s_next, r, done, _ = self.env.step(a)
                rew += r

                # Push a transition
                self.memory.push((s, a, r, s_next, done))
                s = s_next
                self.loss = self.train()

            self.episode_durations.append(episode_duration)
            self.rewards.append(r)
            if r == -1:
                self.num_deaths += 1
            self.number_of_deaths.append(self.num_deaths)

            if self.episode_number % self.save_every == 0:
                self.save_model()

            if self.episode_number % self.plot_every == 0:
                self.plot_results()


    def save_model(self):

        file_path = '{}/{}_ep={}.pt'.format(self.exp_folder, self.experiment_name, self.episode_number)
        torch.save({
            'epoch': self.episode_number,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'rewards': self.rewards,
            'episode_durations': self.episode_durations,
            'number_deaths': self.num_deaths,
            'deaths': self.number_of_deaths
        }, file_path)

    def load_model(self, episode=False):
        if episode:
            file_path = '{}/{}_ep={}.pt'.format(self.exp_folder, self.experiment_name, episode)
            self.episode_number = episode
        else:
            file_path = '{}/{}_ep={}.pt'.format(self.exp_folder, self.experiment_name, self.episode_number)
        checkpoint = torch.load(file_path)

        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']
        self.num_deaths = checkpoint['number_deaths']
        self.episode_durations = checkpoint['episode_durations']
        self.rewards = checkpoint['rewards']
        self.number_of_deaths = checkpoint['deaths']

    def plot_results(self):
        fig = plt.figure()
        plt.plot(smooth(self.episode_durations, self.smooth_factor))
        plt.title('Episode durations per episode')
        fig.savefig('{}/{}_ep_dur.png'.format(self.fig_folder, self.experiment_name))

        fig = plt.figure()
        plt.plot(smooth(self.rewards, self.smooth_factor))
        plt.title('Reward per episode')
        fig.savefig('{}/{}_rewards.png'.format(self.fig_folder, self.experiment_name))

        fig = plt.figure()
        plt.plot(self.number_of_deaths)
        plt.title('Number of deaths over time')
        fig.savefig('{}/{}_nem_deaths.png'.format(self.fig_folder, self.experiment_name))

        print('The current number of deaths is {} after {} episodes'.format(self.num_deaths, self.episode_number))
        plt.close()

def run_Q_learner(network, dynamic_holes, gridworld_size, i):
    Trainer = trainer_Q_network(network=network, dynamic_holes=dynamic_holes, gridworld_size=gridworld_size)
    Trainer.run_episodes()

    fn = 'big_chart_pickles/{}_{}_{}.pt'.format(gridworld_size, network.__class__.__name__, datetime.datetime.now().timestamp())

    with open(fn, "wb") as pf:
        pickle.dump((Trainer.rewards, gridworld_size, network.__class__.__name__, Trainer.episode_durations), pf)

    'Finished a Q learner for {} of size {}, this is number {}'.format(network.__class__.__name__, gridworld_size, i)


def google_experiment(network, dynamic_holes, number_of_epochs):
    Trainer = trainer_Q_network(network=network, dynamic_holes=dynamic_holes, num_episodes=number_of_epochs, save_every=1000, plot_every=500)
    Trainer.run_episodes()


if __name__ == "__main__":


    # dynamic_holes_poss = [True, False]
    # dynamic_start_poss = [True, False]
    # # # network_poss = [QNetwork, QNetwork_deeper, DQN]
    # network_poss = [QNetwork, SimpleCNN]
    # gridworld_sizes = [x for x in range(3,33)]
    # # embeddings = [True, False]
    # number_of_experiments = 10
    #
    # Parallel(n_jobs=4)(delayed(run_Q_learner) (network, False, size, i) for network in network_poss for size in gridworld_sizes for i in range(number_of_experiments))
    # trainer_Q_network(network=SimpleCNN, dynamic_holes=True, dynamic_start=False)

    dynamic_holes_poss = [True, False]
    network_poss = [DQN, SimpleCNN]
    Parallel(n_jobs=4)(
        delayed(google_experiment) (network, True, 1000000) for network in network_poss)


import numpy as np
import gym
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Convolution2D
from keras.optimizers import Adam
from keras.losses import logcosh
from keras.callbacks import TensorBoard, ModelCheckpoint

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
import matplotlib.pyplot as plt
from MarioGym import MarioGym
import os


# Get the environment and extract the number of actions.
env = MarioGym()
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n
tb_log_dir = 'logs/tmp/{}'.format(time.time())
tb_callback = TensorBoard(log_dir=tb_log_dir, batch_size=32, write_grads=True, write_images=True)
cp = ModelCheckpoint('logs/cp/checkpoint-{episode_reward:.2f}-{epoch:02d}-.h5f', monitor='episode_reward', verbose=0, save_best_only=False, save_weights_only=True, mode='max', period=1)

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape = (1,6)))
model.add(Dense(32))
model.add(Activation('tanh'))
model.add(Dense(32))
model.add(Activation('tanh'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=500000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.99, value_min=.1, value_test=.1, nb_steps=2000000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10000,
               target_model_update=0.001, policy=policy, train_interval=1, enable_dueling_network=False, delta_clip=1.0)
dqn.compile(Adam(lr=1e-5), metrics=['mae'])

dqn.load_weights('logs/cp/checkpoint-129.07-299-.h5f')

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=5000000, visualize=True, verbose=2, callbacks=[tb_callback, cp])

# After training is done, we save the final weights.
#dqn.save_weights('dqn_{}_weights.h5f'.format('mario'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True, action_repetition=5)

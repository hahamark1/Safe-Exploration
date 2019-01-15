import numpy as np
import gym
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
from keras.losses import logcosh
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras.backend as K
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from MarioGym import MarioGym
import os


# Get the environment and extract the number of actions.
env = MarioGym()
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
tb_log_dir = 'logs/tmp/{}'.format(time.time())
tb_callback = TensorBoard(log_dir=tb_log_dir, batch_size=32, write_grads=True, write_images=True)
cp = ModelCheckpoint('logs/cp/checkpoint-{episode_reward:.2f}-{epoch:02d}-.h5f', monitor='episode_reward', verbose=0, save_best_only=False, save_weights_only=True, mode='max', period=25000)
INPUT_SHAPE = (40, 80)
WINDOW_LENGTH=4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 2  # (height, width, channel)
        return observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch



input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')

model.add(Convolution2D(8, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(16, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(16, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

processor = AtariProcessor()
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=4)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.99, value_min=.1, value_test=.1, nb_steps=1000000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10000, processor=processor,
               target_model_update=10000, policy=policy, train_interval=4, enable_dueling_network=True, delta_clip=1.0)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

#dqn.load_weights('logs/cp/checkpoint-129.07-299-.h5f')

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=5000000, visualize=True, verbose=2, callbacks=[tb_callback, cp])

# After training is done, we save the final weights.
#dqn.save_weights('dqn_{}_weights.h5f'.format('mario'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True, action_repetition=5)

import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras.backend as K
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from MarioGym import MarioGym


# Get the environment and extract the number of actions.
env = MarioGym(headless=True)
np.random.seed()
env.seed()
nb_actions = env.action_space.n
tb_log_dir = 'logs/tmp/{}'.format(time.time())
tb_callback = TensorBoard(log_dir=tb_log_dir, batch_size=32, write_grads=True, write_images=True)
cp = ModelCheckpoint('logs/cp/checkpoint-{episode_reward:.2f}-{epoch:02d}-.h5f', monitor='episode_reward', verbose=0, save_best_only=False, save_weights_only=True, mode='max', period=500)
INPUT_SHAPE = (40, 80)

# HYPERPARAMETERS
TRAINING_STEPS = 2000
WINDOW_LENGTH = 4
REPLAY_MEMORY = 500000
MAX_EPSILON = 0.99
MIN_EPSILON = 0.01
EPSILON_DECAY_PERIOD = 0.75
LEARNING_RATE = 0.00025
DUELING = True
WARMUP_STEPS = 200000
TARGET_MODEL_UPDATE = 10000
DELTA_CLIP = 1.0
ACTION_REPETITION = 5

class AtariProcessor(Processor):
    def process_observation(self, observation):
        return observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32')
        return processed_batch


# DEFINE THE MODEL
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

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
processor = AtariProcessor()

memory = SequentialMemory(limit=REPLAY_MEMORY,
                          window_length=WINDOW_LENGTH)

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps', value_max=MAX_EPSILON,
                              value_min=MIN_EPSILON,
                              value_test=MIN_EPSILON,
                              nb_steps=int(TRAINING_STEPS*EPSILON_DECAY_PERIOD))

dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               memory=memory,
               nb_steps_warmup=WARMUP_STEPS,
               processor=processor,
               target_model_update=TARGET_MODEL_UPDATE,
               policy=policy,
               train_interval=WINDOW_LENGTH,
               enable_dueling_network=DUELING,
               delta_clip=DELTA_CLIP,
               )

dqn.compile(Adam(lr=LEARNING_RATE),
            metrics=['mae'])

#dqn.load_weights('logs/cp/checkpoint-129.07-299-.h5f')

# Okay, now it's time to learn something!
dqn.fit(env,
        nb_steps=TRAINING_STEPS,
        visualize=True,
        verbose=2,
        callbacks=[cp],
        action_repetition=ACTION_REPETITION)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format('mario'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True, action_repetition=ACTION_REPETITION)

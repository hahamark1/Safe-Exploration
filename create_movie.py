from MarioGym import MarioGym
import matplotlib.pyplot as plt
import random

import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)

from DQN_SE import VALID_ACTIONS, make_boltzmann_policy, make_epsilon_greedy_policy, Estimator, StateProcessor

from video_files import *

def replace_checkpoint(file_path):

    with open(file_path, 'w') as fp:
        fp.write('model_checkpoint_path: "{}/model"'.format('/'.join(file_path.split('/')[0:-1])))
        fp.write('all_model_checkpoint_paths: "{}/model"'.format('/'.join(file_path.split('/')[0:-1])))


EXPERIMENT_NAME = 'safe_exploration_4.0'
HEADLESS = False
LEVEL_NAME = 'Level-basic-one-hole.json'
epsilon = 0.1

POLICY = 'GREEDY'
WINDOW_LENGTH = 4


def make_experiment_video(experiment_name, experiment_dir):
    # Get the environment and extract the number of actions.
    env = MarioGym(headless=HEADLESS, level_name=LEVEL_NAME, step_size=1)
    tf.reset_default_graph()
    global_step = tf.Variable(0, name='global_step', trainable=False)


    # Create directories for checkpoints and summaries
    # experiment_dir = os.path.abspath("./experiments/{}".format(experiment_name))
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    video_path = os.path.join(experiment_dir, "video", experiment_name)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    saver = tf.train.Saver()

    # Load a previous checkpoint if we find one
    print(checkpoint_dir)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
    state_processor = StateProcessor()


    restart = False
    plt.ion()

    image_counter = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)

        # The policy we're following
        if POLICY == 'BOLTZMAN':
            policy = make_boltzmann_policy(
                q_estimator,
                len(VALID_ACTIONS))
        else:
            policy = make_epsilon_greedy_policy(q_estimator, len(VALID_ACTIONS))

        total_state = env.reset(levelname=LEVEL_NAME)
        state = state_processor.process(sess, total_state, 1)
        state = np.stack([state] * WINDOW_LENGTH, axis=2)
        total_state = np.stack([state], axis=2)

        while not restart:
            # save the current screen
            make_video(env.screen, image_counter, video_path)

            # Perform the next action given the policy learnt
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_total_state, reward, restart, info = env.step(VALID_ACTIONS[action])

            next_state = state_processor.process(sess, next_total_state, 1)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

            next_total_state = np.stack([next_state], axis=2)

            image_counter += 1

            state = next_state
            total_state = next_total_state

    create_video_from_images(video_path, experiment_name)

rootdir = '/home/hahamark/Desktop/experiments'

for subdir, dirs, files in os.walk(rootdir, topdown=False):
    for file in files:
        if subdir.split('/')[-2].startswith('safe_exploration'):
            experiment_name = subdir.split('/')[-2]
            expiriment_path = '/'.join(subdir.split('/')[0:-1])
            if file == 'checkpoint':
                replace_checkpoint(os.path.join(subdir, file))
                print('Now making a video for experiment: {}'.format(experiment_name))
                make_experiment_video(experiment_name, expiriment_path)
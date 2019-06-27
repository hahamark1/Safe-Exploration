from MarioGym import MarioGym
import matplotlib.pyplot as plt
import random

import numpy as np
import os
import skimage
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)

from DQN_SE import VALID_ACTIONS, make_boltzmann_policy, make_epsilon_greedy_policy, Estimator, StateProcessor

from video_files import *
from constants import *

def replace_checkpoint(file_path):

    with open(file_path, 'w') as fp:
        fp.write('model_checkpoint_path: "{}/model"'.format('/'.join(file_path.split('/')[0:-1])))
        fp.write('all_model_checkpoint_paths: "{}/model"'.format('/'.join(file_path.split('/')[0:-1])))


def empty_folder(fp):
    for the_file in os.listdir(fp):
        file_path = os.path.join(fp, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def get_mask(center, size, r):
   y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
   keep = x*x + y*y <= 1
   mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
   mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
   return mask/mask.max()



def make_saliency(action_probs, policy, sess, state, epsilon,env,image_counter, video_path, saliency_path):
   # state is 128x128x4 (4 history frames and all frames are black/white
   # values of the states correspond to ??
   r = 2 #radius
   d = 2 #density
   saliency = np.zeros((int(IMAGE_SIZE / d) + 1, int(IMAGE_SIZE / d) + 1))  # saliency scores S(t,i,j)
   # add blur to every i,j and check resulting action
   for i in range(0, IMAGE_SIZE, d):
       for j in range(0, IMAGE_SIZE, d):
           mask = get_mask(center=[i, j], size=[IMAGE_SIZE, IMAGE_SIZE], r=r)
           state = state.astype(np.float32)
           state_mask = state
           state_mask[0,:,:] += mask

           action_probs_s = policy(sess, state_mask, epsilon)
           diff = action_probs - action_probs_s
           saliency[int(i / d), int(j / d)] = 0.5*sum(diff*diff)
   pmax = saliency.max()
   saliency = skimage.transform.resize(saliency, (640, 480)).astype(np.float32)

   # create images
   saliency = saliency.astype('uint8')
   imgdata = pygame.surfarray.array3d(env.screen)
   imgdata.swapaxes(0, 1)
   imgdata = imgdata.astype('uint8')
   imgdata[:, :, 0] += saliency
   env_mario = pygame.surfarray.make_surface(imgdata)
   env_empty = pygame.surfarray.make_surface(saliency)

   make_video(env_mario, image_counter, video_path)
   make_video(env_empty, image_counter, saliency_path)



rootdir = '/home/hahamark/Documents/AI/thesis/moral-mario/experiments/safe_exploration_6.3/'


def make_experiment_video(experiment_name, experiment_dir):
    # Get the environment and extract the number of actions.
    env = MarioGym(HEADLESS, step_size=STEP_SIZE_VIDEO, level_name=LEVEL_NAME, partial_observation=PARTIAL_OBSERVATION, distance_reward=DISTANCE_REWARD)

    tf.reset_default_graph()
    global_step = tf.Variable(0, name='global_step', trainable=False)


    # Create directories for checkpoints and summaries
    # experiment_dir = os.path.abspath("./experiments/{}".format(experiment_name))
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")

    video_path = os.path.join(experiment_dir, "video", experiment_name)
    saliency_path = os.path.join(experiment_dir, "saliency", experiment_name)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    if not os.path.exists(saliency_path):
        os.makedirs(saliency_path)

    empty_folder(video_path)
    empty_folder(saliency_path)

    saver = tf.train.Saver()

    # Load a previous checkpoint if we find one
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

        # Reset the environment
        total_state = env.reset(levelname=LEVEL_NAME)
        state = state_processor.process(sess, total_state, 1)
        state = np.stack([state] * WINDOW_LENGTH, axis=0)
        total_state = np.stack([state], axis=0)

        while not restart:

            # Perform the next action given the policy learnt
            action_probs = policy(sess, state, epsilon)

            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            if SALIENCY:
                # create screen with saliency
                make_saliency(action_probs, policy, sess, state, epsilon, env, image_counter, video_path,
                                         saliency_path)
                image_counter += 1
            else:
                make_video(env.screen, image_counter, video_path)
                image_counter += 1

            next_total_state, reward, restart, info = env.step(VALID_ACTIONS[action])

            next_state = state_processor.process(sess, next_total_state, 1)
            next_state = np.append(state[1:,:,:], np.expand_dims(next_state, 0), axis=0)

            next_total_state = np.stack([next_state], axis=0)



            state = next_state
            total_state = next_total_state

    create_video_from_images(video_path, experiment_name)



for subdir, dirs, files in os.walk(rootdir, topdown=False):
    for file in files:
        if subdir.split('/')[-2].startswith('safe_exploration'):
            experiment_name = subdir.split('/')[-2]
            experiment_path = '/'.join(subdir.split('/')[0:-1])
            if file == 'checkpoint':
                replace_checkpoint(os.path.join(subdir, file))
                print('Now making a video for experiment: {}'.format(experiment_name))
                make_experiment_video(experiment_name, experiment_path)
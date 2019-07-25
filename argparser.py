import argparse
from constants import *

parser = argparse.ArgumentParser()

parser.add_argument('-experiment_type', default=EXPERIMENT, dest='EXPERIMENT',
                    help='Store the experiment type')

parser.add_argument('-image_size', default=IMAGE_SIZE, dest='IMAGE_SIZE',
                    help='Store the image input size')
parser.add_argument('-headless', action='store_true', default=HEADLESS, dest='HEADLESS',
                    help='Store the headless bool')
parser.add_argument('-experiment_name', default=EXPERIMENT_NAME, dest='EXPERIMENT_NAME',
                    help='Store the experiment name')
parser.add_argument('-level_name', default=LEVEL_NAME, dest='LEVEL_NAME',
                    help='Store the level name')
parser.add_argument('-partial_observation', action='store_true', default=PARTIAL_OBSERVATION, dest='PARTIAL_OBSERVATION',
                    help='Store the partial observation bool')
parser.add_argument('-distance_reward', action='store_true', default=DISTANCE_REWARD, dest='DISTANCE_REWARD',
                    help='Store the distance reward bool')
parser.add_argument('-policy', default=POLICY, dest='POLICY',
                    help='Store the policy, either GREEDY or BOLTZMAN')
parser.add_argument('-episode_length', default=EPISODE_LENGTH, dest='EPISODE_LENGTH',
                    help='Store the episode length')
parser.add_argument('-step_size', default=STEP_SIZE, dest='STEP_SIZE',
                    help='Store the step_size')

parser.add_argument('-hole_reward', default=HOLE_REWARD, dest='HOLE_REWARD',
                    help='Store the absolute hole reward value')
parser.add_argument('-coin_reward', default=COIN_REWARD, dest='COIN_REWARD',
                    help='Store the coin reward value')

parser.add_argument('-replay_memory_size', default=REPLAY_MEMORY_SIZE, dest='REPLAY_MEMORY_SIZE',
                    help='Store the replay memory size')
parser.add_argument('-epsilon_decay_steps', default=EPSILON_DECAY_STEPS, dest='EPSILON_DECAY_STEPS',
                    help='Store the epsilon decay steps value')

parser.add_argument('-saliency', action='store_true', default=SALIENCY, dest='SALIENCY',
                    help='Store the saliency bool')
parser.add_argument('-use_memory', action='store_true', default=USE_MEMORY, dest='USE_MEMORY',
                    help='Store the demonstration memory bool')
parser.add_argument('-prioritize_replay', action='store_true', default=PRIORITIZE_MEMORY, dest='PRIORITIZE_MEMORY',
                    help='Store the prioritize replay bool')
parser.add_argument('-min_epsilon', default=MIN_EPSILON, dest='MIN_EPSILON',
                    help='Store the minimal epsilon')

parser.add_argument('-epsilon_video', default=epsilon, dest='epsilon',
                    help='Store the video epsilon')


parser.add_argument('-learning_rate', default=LR, dest='LR',
                    help='Store the Learning Rate')
parser.add_argument('-weight_decay', default=WEIGHT_DECAY, dest='WEIGHT_DECAY',
                    help='Store the weight decay value')
parser.add_argument('-momentum', default=MOMENTUM, dest='MOMENTUM',
                    help='Store the momentum value')
parser.add_argument('-epsilon_network', default=Epsilon_network, dest='Epsilon_network',
                    help='Store the epsilon network value')
parser.add_argument('-discount_factor', default=DISCOUNT_FACTOR, dest='DISCOUNT_FACTOR',
                    help='Store the discount factor value')
parser.add_argument('-batch_size', default=BATCH_SIZE, dest='BATCH_SIZE',
                    help='Store the batch_size value')
parser.add_argument('-update_target_step', default=UPDATE_TARGET_STEP, dest='UPDATE_TARGET_STEP',
                    help='Store the update target step value')
import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import pickle

from datetime import datetime

from GridworldCoinSharingGym import GridworldCoinSharingGym
from GridworldTest import GridworldTest
from GridworldGoombaGym import GridworldGoombaGym
import tensorflow as tf



if "../" not in sys.path:
  sys.path.append("../")

from lib import plotting
from collections import deque, namedtuple


seed = 238
np.random.seed(seed)
random.seed(seed)



VALID_ACTIONS = [0, 1, 2, 3, 4]
WINDOW_LENGTH = 1
INPUT_DIM = 5

class StateProcessor():
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[INPUT_DIM, INPUT_DIM, 2], dtype=tf.float64)
            self.output1 = self.input_state[:,:,0]
            self.output2 = self.input_state[:,:,1]


    def process(self, sess, state, output):
        if output == 1:
            return sess.run(self.output1, {self.input_state: state})
        elif output == 2:
            return sess.run(self.output2, {self.input_state: state})

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        tf.set_random_seed(seed)

        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are WINDOW_LENGTH RGB frames of shape 160, 160 each
        self.X_pl = tf.placeholder(shape=[None, 5, 5, WINDOW_LENGTH], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 100
        batch_size = tf.shape(self.X_pl)[0]


        # # Three convolutional layers
        #conv1 = tf.contrib.layers.conv2d(
        #     X, 32, 3, 1, padding='SAME', activation_fn=tf.nn.relu)
        # conv2 = tf.contrib.layers.conv2d(
        #     conv1, 32, 3, 1, padding='SAME', activation_fn=tf.nn.relu)
        # conv3 = tf.contrib.layers.conv2d(
        #     conv2, 64, 3, 1, padding='SAME', activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(X)
        fc1 = tf.contrib.layers.fully_connected(flattened, 128)
        fc2 = tf.contrib.layers.fully_connected(fc1, 128)

        self.predictions = tf.contrib.layers.fully_connected(fc2, len(VALID_ACTIONS), activation_fn=None)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])


    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, WINDOW_LENGTH, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, WINDOW_LENGTH, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        print(f'q_values: {q_values}')
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def log_scalar(tag, value, global_step, writer):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    writer.add_summary(summary, global_step=global_step)
    writer.flush()

def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    empathic_estimator,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50,
                    selfishness=0.5):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []

    coins_collected = [0]
    enemy_coins_collected = [0]
    enemy_rewards = [0]
    got_killed = [0]
    enemies_killed = [0]
    equalities = []
    rewards = [0]
    episode_steps = []

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_kills=np.zeros(num_episodes))

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")
    pickle_path = os.path.join(experiment_dir, "pickles")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        empathic_estimator,
        len(VALID_ACTIONS))

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    total_state = env.reset()
    state = state_processor.process(sess, total_state, 1)
    enemy_state = state_processor.process(sess, total_state, 2)

    state = np.stack([state] * WINDOW_LENGTH, axis=2)
    enemy_state = np.stack([enemy_state] * WINDOW_LENGTH, axis=2)

    total_state = np.stack([state, enemy_state], axis=2)

    for i in range(replay_memory_init_size):
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        next_total_state, reward, done, info = env.step(VALID_ACTIONS[action])

        next_state = state_processor.process(sess, next_total_state, 1)
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

        next_enemy_state = state_processor.process(sess, next_total_state, 2)
        next_enemy_state = np.append(enemy_state[:, :, 1:], np.expand_dims(next_enemy_state, 2), axis=2)

        next_total_state = np.stack([next_state, next_enemy_state], axis=2)

        replay_memory.append(Transition(total_state, action, reward, next_total_state, done))
        if done:
            total_state = env.reset()
            state = state_processor.process(sess, total_state, 1)
            enemy_state = state_processor.process(sess, total_state, 2)

            state = np.stack([state] * WINDOW_LENGTH, axis=2)
            enemy_state = np.stack([enemy_state] * WINDOW_LENGTH, axis=2)

            total_state = np.stack([state, enemy_state], axis=2)
        else:
            total_state = next_total_state
            state = next_state
            enemy_state = next_enemy_state


    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        # Reset the environment
        total_state = env.reset()
        state = state_processor.process(sess, total_state, 1)
        enemy_state = state_processor.process(sess, total_state, 2)

        state = np.stack([state] * WINDOW_LENGTH, axis=2)
        enemy_state = np.stack([enemy_state] * WINDOW_LENGTH, axis=2)

        total_state = np.stack([state, enemy_state], axis=2)

        loss = None

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            q_estimator.summary_writer.add_summary(episode_summary, total_t)

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            #print("Step {} ({}) @ Episode {}/{}, loss: {}".format(
            #        t, total_t, i_episode + 1, num_episodes, loss))
            #sys.stdout.flush()

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_total_state, reward, done, info = env.step(VALID_ACTIONS[action])

            next_state = state_processor.process(sess, next_total_state, 1)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

            next_enemy_state = state_processor.process(sess, next_total_state, 2)
            next_enemy_state = np.append(enemy_state[:, :, 1:], np.expand_dims(next_enemy_state, 2), axis=2)

            next_total_state = np.stack([next_state, next_enemy_state], axis=2)

            coins_collected[-1] += info['coins_collected']
            enemy_coins_collected[-1] += info['enemy_coins_collected']
            enemy_rewards[-1] += info['enemy_reward']
            got_killed[-1] += info['got_killed']
            enemies_killed[-1] += info['num_killed']
            rewards[-1] += reward

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(total_state, action, reward, next_total_state, done))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            stats.episode_kills[i_episode] += info['num_killed']

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets (Double DQN)
            q_values_next = q_estimator.predict(sess, next_states_batch[:,:,:,0,:])
            #q_values_enemy_next = q_estimator.predict(sess, next_states_batch[:,:,:,1,:])

            best_actions = np.argmax(q_values_next, axis=1)
            #best_enemy_actions = np.argmax(q_values_enemy_next, axis=1)


            q_values_next_target = target_estimator.predict(sess, next_states_batch[:,:,:,0,:])
            q_values_next_enemy_target = target_estimator.predict(sess, next_states_batch[:,:,:,1,:])

            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                            (discount_factor * q_values_next_target[np.arange(batch_size), best_actions])

            targets_enemy = discount_factor * np.mean(q_values_next_enemy_target, axis=1)

            targets_empathy = selfishness*targets_batch + (1-selfishness)*targets_enemy

            # Perform gradient descent update
            states_batch = np.array(states_batch)

            loss1 = q_estimator.update(sess, states_batch[:,:,:,0,:], action_batch, targets_batch)
            loss2 = empathic_estimator.update(sess, states_batch[:,:,:,0,:], action_batch, targets_empathy)

            if done:
                equality = (2*min(info['total_enemy_reward'], info['total_reward'])) / (info['total_enemy_reward'] + info['total_reward'])
                equalities.append(equality)
                episode_steps.append(t)
                print(f'enemy_rewards: {enemy_rewards[-1]}, reward: {stats.episode_rewards[i_episode]}, equality: {equality}')
                log_scalar('coins_collected', coins_collected[-1], total_t, q_estimator.summary_writer)
                log_scalar('enemy_coins_collected', enemy_coins_collected[-1], total_t, q_estimator.summary_writer)
                log_scalar('equality',equality, total_t, q_estimator.summary_writer)

                pickle.dump(coins_collected, open(f'{experiment_dir}/pickles/coins_collected.pickle', 'wb'))
                pickle.dump(equalities, open(f'{experiment_dir}/pickles/equality.pickle', 'wb'))
                pickle.dump(enemy_coins_collected, open(f'{experiment_dir}/pickles/enemy_coins_collected.pickle', 'wb'))
                pickle.dump(got_killed, open(f'{experiment_dir}/pickles/got_killed.pickle', 'wb'))
                pickle.dump(enemies_killed, open(f'{experiment_dir}/pickles/enemies_killed.pickle', 'wb'))
                pickle.dump(episode_steps, open(f'{experiment_dir}/pickles/episode_length.pickle', 'wb'))
                pickle.dump(rewards, open(f'{experiment_dir}/pickles/rewards.pickle', 'wb'))

                coins_collected.append(0)
                enemy_coins_collected.append(0)
                enemy_rewards.append(0)
                got_killed.append(0)
                enemies_killed.append(0)
                rewards.append(0)

                total_state = env.reset()
                state = state_processor.process(sess, total_state, 1)
                enemy_state = state_processor.process(sess, total_state, 2)

                state = np.stack([state] * WINDOW_LENGTH, axis=2)
                enemy_state = np.stack([enemy_state] * WINDOW_LENGTH, axis=2)

                total_state = np.stack([state, enemy_state], axis=2)
                break
            else:
                total_state = next_total_state
                state = next_state
                enemy_state = next_enemy_state


            #state = next_state
            total_t += 1

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
        episode_summary.value.add(simple_value=stats.episode_kills[i_episode], node_name="episode_kills", tag="episode_kills")
        q_estimator.summary_writer.add_summary(episode_summary, total_t)
        q_estimator.summary_writer.flush()

        yield total_t, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode+1],
            episode_rewards=stats.episode_rewards[:i_episode+1],
            episode_kills=stats.episode_kills[:i_episode+1])

    return stats


step_reward = 1
dead_reward = 0
kill_reward = 0
max_steps = 500
selfishness = 1.0

tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath(f"logs/goomba/version_10.7/maxsteps_{max_steps}/step_reward_{step_reward}/dead_reward_{dead_reward}/kill_reward_{kill_reward}/selfishness_{selfishness}/seed_{seed}")

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")
empathic_estimator = Estimator(scope="empathic")

# State processor
state_processor = StateProcessor()


env = GridworldGoombaGym(headless=True, step_reward=step_reward, dead_reward=dead_reward, kill_reward=kill_reward, max_steps=max_steps, gridworld_size=7, seed=seed)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t, stats in deep_q_learning(sess,
                                    env,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    empathic_estimator=empathic_estimator,
                                    state_processor=state_processor,
                                    experiment_dir=experiment_dir,
                                    num_episodes=200000,
                                    replay_memory_size=500000,
                                    replay_memory_init_size=50000,
                                    update_target_estimator_every=10000,
                                    epsilon_start=1.0,
                                    epsilon_end=0.01,
                                    epsilon_decay_steps=1000000,
                                    discount_factor=0.99,
                                    batch_size=32,
                                    selfishness=selfishness):

        print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))

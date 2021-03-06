import numpy as np
from ou_noise import OUNoise
from critic import CriticNetwork
from actor import ActorNetwork
from grad_inverter import GradInverter
import tensorflow as tf
from data_manager import DataManager

import prio_data_manager

# For saving replay buffer
import os
import time

# Visualization
from state_visualizer import CostmapVisualizer


# How big are our mini batches
BATCH_SIZE = 32

# How big is our discount factor for rewards
GAMMA = 0.99

# How does our noise behave (MU = Center value, THETA = How strong is noise pulled to MU, SIGMA = Variance of noise)
MU = 0.0
THETA = 0.1
SIGMA = 0.1

# Action boundaries
A0_BOUNDS = [-0.3, 0.3]
A1_BOUNDS = [-0.3, 0.3]

# Should we load a saved net
PRE_TRAINED_NETS = False

# data path where experiences, saved networks, and tf logs are stored
DATA_PATH = '/media/nutzer/D478693978691C0C/RL_nav_data'
#DATA_PATH = os.path.expanduser('~') + '/RL_nav_data'
# os.path.join(os.path.dirname(__file__), os.pardir)

# If we use a pretrained net
NET_SAVE_PATH = DATA_PATH + "/pre_trained_networks/pre_trained_networks"
NET_LOAD_PATH = DATA_PATH + "/pre_trained_networks/pre_trained_networks-600000"

# If we don't use a pretrained net we should load pre-trained filters from this path
FILTER_LOAD_PATH = DATA_PATH + "/pre_trained_filters/pre_trained_filters"

# path to tensorboard data
TFLOG_PATH = DATA_PATH + '/tf_logs'

# path to experience files
EXPERIENCE_PATH = DATA_PATH + '/experiences'

# Visualize an initial state batch for debugging
VISUALIZE_BUFFER = False

# How often are we saving the net
SAVE_STEP = 50000


class DDPG:

    def __init__(self):

        # Initialize our session
        self.session = tf.Session()
        self.graph = self.session.graph

        with self.graph.as_default():

            # View the state batches
            self.visualize_input = VISUALIZE_BUFFER
            if self.visualize_input:
                self.viewer = CostmapVisualizer()

            # Hardcode input size and action size
            self.height = 86
            self.width = self.height
            self.depth = 4
            self.action_dim = 2

            # Initialize the current action and the old action and old state for setting experiences
            self.old_state = np.zeros((self.width, self.height, self.depth), dtype='int8')
            self.old_action_output = np.zeros(2, dtype='float')
            self.net_q_value = np.zeros(1, dtype='float')
            self.q_value = np.zeros(1, dtype='float')
            self.old_q_value = np.zeros(1, dtype='float')
            self.network_action = np.zeros(2, dtype='float')
            self.noise_action = np.zeros(2, dtype='float')
            self.action_output = np.zeros(2, dtype='float')
            self.net_action = np.zeros(2, dtype='float')

            # Initialize the grad inverter object to keep the action bounds
            self.grad_inv = GradInverter(A0_BOUNDS, A1_BOUNDS, self.session)

            # Initialize summary writers to plot variables during training
            self.summary_op = tf.merge_all_summaries()
            self.summary_writer = tf.train.SummaryWriter(TFLOG_PATH)

            # Initialize actor and critic networks
            self.actor_network = ActorNetwork(self.height, self.action_dim, self.depth, self.session,
                                              self.summary_writer)
            self.critic_network = CriticNetwork(self.height, self.action_dim, self.depth, self.session,
                                                self.summary_writer)

            # Initialize the saver to save the network params
            self.saver = tf.train.Saver()

            # initialize the experience data manger
            self.data_manager = DataManager(BATCH_SIZE, EXPERIENCE_PATH, self.session)
            #self.prio_data_manager = prio_data_manager.DataSet(random_state = None, max_size= 1000000,
            #                                                   use_priority= True)

            # Should we load the pre-trained params?
            # If so: Load the full re-trained net
            # Else:  Initialize all variables the overwrite the conv layers with the pretrained filters
            if PRE_TRAINED_NETS:
                self.saver.restore(self.session, NET_LOAD_PATH)#
                print "restored net"
            else:
                print "loading filters"
                self.session.run(tf.initialize_all_variables())
                self.critic_network.restore_pretrained_weights(FILTER_LOAD_PATH)
                self.actor_network.restore_pretrained_weights(FILTER_LOAD_PATH)
                print "done"
            tf.train.start_queue_runners(sess=self.session)
            time.sleep(1)

            # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
            self.exploration_noise = OUNoise(self.action_dim, MU, THETA, SIGMA)
            self.noise_flag = True

            # Initialize time step
            self.training_step = 0

            # Flag: don't learn the first experience
            self.first_experience = True

            # After the graph has been filled add it to the summary writer
            self.summary_writer.add_graph(self.graph)

    def train(self):

        # Check if the buffer is big enough to start training
        if self.data_manager.enough_data():

            # TODO: get batch from prio manager
            # get the next random batch from the data manger
            state_batch, \
                action_batch, \
                reward_batch, \
                next_state_batch, \
                is_episode_finished_batch = self.data_manager.get_next_batch()

            state_batch = np.divide(state_batch, 100.0)
            next_state_batch = np.divide(next_state_batch, 100.0)

            # Are we visualizing the first state batch for debugging?
            # If so: We have to scale up the values for grey scale before plotting
            if self.visualize_input:
                state_batch_np = np.asarray(state_batch)
                state_batch_np = np.multiply(state_batch_np, -100.0)
                state_batch_np = np.add(state_batch_np, 100.0)
                self.viewer.set_data(state_batch_np)
                self.viewer.run()
                self.visualize_input = True

            # Calculate y for the td_error of the critic
            y_batch = []
            next_action_batch = self.actor_network.target_evaluate(next_state_batch)
            q_value_batch = self.critic_network.target_evaluate(next_state_batch, next_action_batch)
            for i in range(0, BATCH_SIZE):
                if is_episode_finished_batch[i]:
                    y_batch.append([reward_batch[i]])
                else:
                    y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])

            # Now that we have the y batch lets train the critic
            self.critic_network.train(y_batch, state_batch, action_batch)

            # Get the action batch so we can calculate the action gradient with it
            # Then get the action gradient batch and adapt the gradient with the gradient inverting method
            action_batch_for_gradients = self.actor_network.evaluate(state_batch)
            q_gradient_batch = self.critic_network.get_action_gradient(state_batch, action_batch_for_gradients)
            q_gradient_batch = self.grad_inv.invert(q_gradient_batch, action_batch_for_gradients)

            # Now we can train the actor
            self.actor_network.train(q_gradient_batch, state_batch)

            # Save model if necessary
            if self.training_step > 0 and self.training_step % SAVE_STEP == 0:
                self.saver.save(self.session, NET_SAVE_PATH, global_step=self.training_step)

            # Update time step
            self.training_step += 1
        # check if we need to enqueue new files to the filename queue of the data manager
        self.data_manager.check_for_enqueue()

    def get_action(self, state):

        # normalize the state input
        state = np.divide(state, 100.0)

        # Get the action
        self.net_action = self.actor_network.get_action(state)
        print "net act", self.net_action

        # Are we using noise?
        if self.noise_flag:
            self.action_output = self.net_action + self.exploration_noise.noise()
            # if action value lies outside of action bounds, rescale the action vector
            if self.action_output[0] < A0_BOUNDS[0] or self.action_output[0] > A0_BOUNDS[1]:
                self.action_output = self.action_output*np.fabs(A0_BOUNDS[0]/self.action_output[0])
            if self.action_output[1] < A0_BOUNDS[0] or self.action_output[1] > A0_BOUNDS[1]:
                self.action_output = self.action_output*np.fabs(A1_BOUNDS[0]/self.action_output[1])
        else:
            self.action_output = self.net_action

        return self.action_output

    def process_dwa_action(self, dwa_actions):
        self.action_output = dwa_actions

    def set_experience(self, state, reward, is_episode_finished):

        self.q_value = self.critic_network.evaluate([state], [self.action_output])
        # self.net_q_value = self.critic_network.evaluate([state], [self.net_action])
        # Live q value output for this action and state
        self.print_q_value(self.q_value)

        # TODO: + epsilon?
        #priority = np.fabs(self.old_q_value - (reward + GAMMA*self.net_q_value))
        #self.prio_data_manager.addSample(self.old_action, self.old_action, reward, is_episode_finished, priority )

        # Make sure we're saving a new old_state for the first experience of every episode
        if self.first_experience:
            self.first_experience = False
        else:
            self.data_manager.store_experience_to_file(self.old_state, self.old_action_output, reward, state,
                                                       is_episode_finished)

        if is_episode_finished:
            self.first_experience = True
            self.exploration_noise.reset()

        # Safe old state and old action for next experience
        self.old_state = state
        self.old_action_output = self.action_output
        self.old_q_value = self.q_value

    def print_q_value(self, q_value):

        string = "-"
        stroke_pos = 30 * q_value[0][0] + 30
        if stroke_pos < 0:
            stroke_pos = 0
        elif stroke_pos > 60:
            stroke_pos = 60
        print '[' + stroke_pos * string + '|' + (60-stroke_pos) * string + ']', "Q: ", q_value, \
            "\tt: ", self.training_step

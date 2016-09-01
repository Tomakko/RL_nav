#!/usr/bin/env python

import rospy
import numpy as np
from neuro_local_planner_wrapper.msg import Transition
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, Vector3



class ROSHandler:

    def __init__(self, holonomic, dwa_planner):

        self.holonomic_robot = holonomic
        self.use_dwa_planner = dwa_planner
        # Initially assumed Input size, since init is false these values will be updated with the first received msg
        self.__init = False
        self.depth = 4
        self.height = 86
        self.width = 86

        self.state = np.zeros((self.width, self.height, self.depth), dtype='int8')
        self.reward = 0.0
        self.executed_action = np.zeros(2, dtype='float')
        self.is_episode_finished = False

        self.__sub_move_base = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/transition", Transition,
                                                self.transition_callback)
        self.__sub_move_base = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/action", Twist,
                                                self.action_callback)
        self.__sub_setting = rospy.Subscriber("/noise_flag", Bool, self.toggle_noise_callback)
        self.__pub = rospy.Publisher("neuro_deep_planner/action", Twist, queue_size=10)

        self.__new_transition_flag = False
        self.__new_action_flag = False
        self.__new_setting_flag = False
        self.noise_flag = True

    def transition_callback(self, transition_msg):

        # If msg is received for the first time adjust parameters

        if not self.__init:
            self.depth = transition_msg.depth
            self.width = transition_msg.width
            self.height = transition_msg.height
            self.state = np.zeros((self.depth, self.width, self.height), dtype='int8')
            self.__init = True

        # Lets update the new reward
        self.reward = transition_msg.reward

        # Check if episode is done or not
        self.is_episode_finished = transition_msg.is_episode_finished

        # Lets update the new costmap its possible that we need to switch some axes here...
        if not self.is_episode_finished:
            temp_state = np.asarray(transition_msg.state_representation, dtype='int8').reshape(self.depth, self.height,
                                                                                               self.width).swapaxes(1, 2)
            self.state = np.rollaxis(temp_state, 0, 3)
        self.__new_transition_flag = True

    # action_msg contains the action executed by our robot
    def action_callback(self, action_msg):
        if self.holonomic_robot:
            self.executed_action[0] = action_msg.linear.x
            self.executed_action[1] = action_msg.linear.y
        else:
            self.executed_action[0] = action_msg.linear.x
            self.executed_action[1] = action_msg.angular.z
            # We have received a new msg
        self.__new_action_flag = True

    def toggle_noise_callback(self, noise_msg):

        # If msg is received for the first time adjust parameters

        self.noise_flag = noise_msg.data

        # We have received a setting
        self.__new_setting_flag = True

    def publish_ddpg_action(self, action):

        # Generate msg output
        if self.holonomic_robot:
            vel_cmd = Twist(Vector3(action[0], action[1], 0), Vector3(0, 0, 0))
        else:
            vel_cmd = Twist(Vector3(action[0], 0, 0), Vector3(0, 0, action[1]))
        # Send the action back
        self.__pub.publish(vel_cmd)

    def check_new_transition_msg(self):
        if self.__new_transition_flag:
            output = True
            self.__new_transition_flag = False
        else:
            # Return true if new msg arrived only once for every new msg
            output = False
        return output

    def check_new_action_msg(self):
        if self.__new_action_flag:
            output = True
            self.__new_action_flag = False
        else:
            # Return true if new msg arrived only once for every new msg
            output = False
        return output

    def check_new_setting(self):

        # Return true if new msg arrived only once for every new msg
        output = False
        if self.__new_setting_flag:
            output = True
            self.__new_setting_flag = False

        return output

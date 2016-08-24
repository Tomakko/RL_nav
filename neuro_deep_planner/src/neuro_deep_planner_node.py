#!/usr/bin/env python

import rospy
from ros_handler import ROSHandler
from ddpg import DDPG


def main():

    # Initialize the ddpg class which handles neural nets
    agent = DDPG()

    # Initialize the ROS handler that handles communication with ROS
    ros_handler = ROSHandler()

    rospy.init_node("neuro_deep_planner", anonymous=False)
    while not rospy.is_shutdown():

        # If we have a new msg we might have to execute an action and need to put the new experience in the buffer
        if ros_handler.check_new_data_msg():
            if not ros_handler.is_episode_finished:
                # Send back the action to execute
                ros_handler.publish_action(agent.get_action(ros_handler.state))

            # Safe the past state and action + the reward and new state into the replay buffer
            agent.set_experience(ros_handler.state, ros_handler.reward, ros_handler.is_episode_finished)

        elif ros_handler.check_new_setting():

            agent.noise_flag = ros_handler.noise_flag

        else:
            # Train the network!
            agent.train()


if __name__ == '__main__':
    main()

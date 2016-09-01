#!/usr/bin/env python

import rospy
from ros_handler import ROSHandler
from ddpg import DDPG

# holonomic or differential robot
HOLONOMIC = True
DWA_PLANNER = False


def main():

    # Initialize the ddpg class which handles neural nets
    agent = DDPG()

    # Initialize the ROS handler that handles communication with ROS
    ros_handler = ROSHandler(HOLONOMIC, DWA_PLANNER)

    # Init ros node
    rospy.init_node("neuro_deep_planner", anonymous=False)
    while not rospy.is_shutdown():

        # new noise settings?
        if ros_handler.check_new_setting():
            agent.noise_flag = ros_handler.noise_flag

        if DWA_PLANNER:
            # if new dwa action has arrived, pass it to the agent in order to save it as an experience
            if ros_handler.check_new_action_msg():
                agent.process_dwa_action(ros_handler.executed_action)
                # Safe the past state and action + the reward and new state into the replay buffer
                agent.set_experience(ros_handler.state, ros_handler.reward, ros_handler.is_episode_finished)

        else:
            # If we have a new msg we might have to execute an action and need to put the new experience in the buffer
            if ros_handler.check_new_transition_msg():
                if not ros_handler.is_episode_finished:
                    # Send back the action to execute
                    ros_handler.publish_ddpg_action(agent.get_action(ros_handler.state))
                    # Safe the past state and action + the reward and new state into the replay buffer
                    agent.set_experience(ros_handler.state, ros_handler.reward, ros_handler.is_episode_finished)

        # Train the network!
        agent.train()


if __name__ == '__main__':
    main()

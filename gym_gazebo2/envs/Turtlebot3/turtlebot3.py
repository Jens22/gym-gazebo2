import gym
gym.logger.set_level(40) # hide warnings
import time
import numpy as np
import copy
import math
import os
import psutil
import signal
import sys
from scipy.stats import skew
from gym import utils, spaces
from gym_gazebo2.utils import ut_generic, ut_launch, ut_gazebo, general_utils
from gym.utils import seeding
from gazebo_msgs.srv import SpawnEntity
import subprocess
import argparse
import transforms3d as tf3d

# ROS 2
import rclpy


from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
# from gazebo_msgs.srv import SetEntityState, DeleteEntity
from gazebo_msgs.msg import ContactState, ModelState#, GetModelList
from std_msgs.msg import String
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ros2pkg.api import get_prefix_path
from builtin_interfaces.msg import Duration

#launch description
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

class Turtlebot3Env(gym.Env):
    """
    TODO. Define the environment.
    """

    def __init__(self):
        
        """
        Initialize the Turtlebot3 environemnt
        """
        rclpy.init()
        # Launch Turtlebot3 in gazebo
        gazeboLaunchFileDir = os.path.join(get_package_share_directory('turtlebot3_gazebo'),'launch')
        launch_desc = LaunchDescription([
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([gazeboLaunchFileDir,'/turtlebot3_4OG.launch.py']))])
        self.launch_subp = ut_launch.startLaunchServiceProcess(launch_desc)

        # Create the node after the new ROS_DOMAIN_ID is set in generate_launch_description()
        self.node = rclpy.create_node(self.__class__.__name__)

        # class variables
        self._odom_msg = None
        self.max_episode_steps = 1024 #default value, can be updated from baselines
        self.iterator = 0
        self.reset_jnts = True
        self._scan_msg = None

        # Subscribe to the appropriate topics, taking into account the particular robot
        qos = QoSProfile(depth=10)
        self._pub_cmd_vel = self.node.create_publisher(Twist, 'cmd_vel', qos)
        self._sub_odom = self.node.create_subscription(Odometry, 'odom', self.odom_callback, qos)
        self._sub_scan = self.node.create_subscription(LaserScan, 'scan', self.scan_callback, qos)
        self.reset_sim = self.node.create_client(Empty, '/reset_simulation')
        self.unpause = self.node.create_client(Empty, '/unpause_physics')
        self.pause = self.node.create_client(Empty,'/pause_physics')


        self.action_space = spaces.Discrete(13)
        
        len_scan = 24
        high = np.inf*np.ones(len_scan)
        high = np.append(high, [np.inf, np.inf])
        low = 0*np.ones(len_scan)
        low = np.append(low, [-1*np.inf, -1*np.inf])
                  
        self.observation_space = spaces.Box(low, high)
        
        # Seed the environment
        self.seed()

    def odom_callback(self, message):
        """
        Callback method for the subscriber of odometry data
        """
        self._odom_msg = message

    def scan_callback(self, message):
        """
        Callback method for the subscriber of scan data
        """
        self._scan_msg = message

    def set_episode_size(self, episode_size):
        self.max_episode_steps = episode_size

    def take_observation(self):
        """
        Take observation from the environment and return it.
        :return: state.
        """
        # # # # Take an observation
        rclpy.spin_once(self.node)
        odom_message = self._odom_msg  #msg of the callback, 
        scan_message = self._scan_msg

        while scan_message is None or odom_message is None:
            #print("I am waiting for massage")
            rclpy.spin_once(self.node)
            odom_message = self._odom_msg
            scan_message = self._scan_msg
            
        #TODO write function that prepare the scan informations
        lastVelocities = [odom_message.twist.twist.linear.x, odom_message.twist.twist.angular.z]
        lastScans = scan_message.ranges
        done = False
        
        for i, item in enumerate(lastScans):
            if lastScans[i] <= 0.2:
                done = True
            elif lastScans[i] == float('inf') or np.isinf(lastScans[i]):
                lastScans[i] = 4.0
                
        #Set observation to None after it has been read.
        self._odom_msg = None
        self._scan_msg = None

        #TODO what should all in the state?
        state = np.r_[np.reshape(lastScans, -1),
                        np.reshape(lastVelocities, -1)]

        return state, done

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Implement the environment step abstraction. Execute action and returns:
            - action
            - observation
            - reward
            - done (status)
        """
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('/unpause simulation service not available, waiting again...')

        unpause = self.unpause.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self.node, unpause)

        self.iterator+=1
        
        # Execute "action"
        action_list = [-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.3]
        V_CONST = 0.3
        vel_cmd = Twist()
        vel_cmd.linear.x = V_CONST
        vel_cmd.angular.z = action_list[action]
        self._pub_cmd_vel.publish(vel_cmd)

        # Take an observation
        obs, done = self.take_observation()
        
        # Get reward, default is 1
        reward = 1.0
        
        # Calculate if the env has been solved
        if done == False:
            done = bool(self.iterator == self.max_episode_steps)
        
        info = {}
        
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('/pause simulation service not available, waiting again...')

        pause = self.pause.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self.node, pause)
        # Return the corresponding observations, rewards, etc.
        return obs, reward, done, info

    def reset(self):
        """
        Reset the agent for a particular experiment condition.
        """
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('/unpause simulation service not available, waiting again...')

        unpause = self.unpause.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self.node, unpause)
        
        self.iterator = 0

        if self.reset_jnts is True:
            # reset simulation
            while not self.reset_sim.wait_for_service(timeout_sec=1.0):
                self.node.get_logger().info('/reset_simulation service not available, waiting again...')

            reset_future = self.reset_sim.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, reset_future)

        # Take an observation
        obs, done = self.take_observation()

        # Return the corresponding observation
        return obs

    def close(self):
        print("Closing " + self.__class__.__name__ + " environment.")
        self.node.destroy_node()
        parent = psutil.Process(self.launch_subp.pid)
        for child in parent.children(recursive=True):
            child.kill()
        rclpy.shutdown()
        parent.kill()

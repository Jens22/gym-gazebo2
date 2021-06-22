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
import random
from scipy.stats import skew
from gym import utils, spaces
from gym_gazebo2.utils import ut_generic, ut_launch, ut_gazebo #, general_utils
from gym.utils import seeding
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
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
        #rclpy.init() #jw TODO init makes problems if I call gym.make out of another ROS node
        
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
        sdf_file_path = os.path.join(get_package_share_directory("turtlebot3_gazebo"), "models", "turtlebot3_burger", "model.sdf")
        objFile = open(sdf_file_path, mode='r')
        self.xml = objFile.read()
        objFile.close()
        
        
        # Subscribe to the appropriate topics, taking into account the particular robot
        qos = QoSProfile(depth=10)
        self._pub_cmd_vel = self.node.create_publisher(Twist, 'cmd_vel', qos)
        self._sub_odom = self.node.create_subscription(Odometry, 'odom', self.odom_callback, qos)
        self._sub_scan = self.node.create_subscription(LaserScan, 'scan', self.scan_callback, qos)
        self.reset_sim = self.node.create_client(Empty, '/reset_simulation')
        self.unpause = self.node.create_client(Empty, '/unpause_physics')
        self.pause = self.node.create_client(Empty,'/pause_physics')
        self.add_entity = self.node.create_client(SpawnEntity, '/spawn_entity')
        self.delete_entity = self.node.create_client(DeleteEntity, '/delete_entity')
        self.action_space = spaces.Discrete(13)
        
        len_scan = 24
        high = np.inf*np.ones(len_scan)
        high = np.append(high, [np.inf, np.inf])
        low = 0*np.ones(len_scan)
        low = np.append(low, [-1*np.inf, -1*np.inf])
                  
        self.observation_space = spaces.Box(low, high)
        
        #Spawn turtlebot3
        while not self.add_entity.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')
        
        req_spawn = SpawnEntity.Request()
        req_spawn.name = 'Turtlebot3'
        req_spawn.xml = self.xml
        req_spawn.robot_namespace = ''
        req_spawn.initial_pose.position.x = -1.0
        req_spawn.initial_pose.position.y = -1.0
        req_spawn.initial_pose.position.z = 0.01
        req_spawn.reference_frame = 'world'
        
        spawn_future = self.add_entity.call_async(req_spawn)
        rclpy.spin_until_future_complete(self.node, spawn_future)
        
        
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
        start_pose_list = []
        pose_1 = Pose()
        pose_1.position.x = -1.0
        pose_1.position.y = 7.0
        pose_1.position.z = 0.01
        pose_1.orientation.x = 0.0
        pose_1.orientation.y = 0.0
        pose_1.orientation.z = -0.7071 
        pose_1.orientation.w = 0.7071
        start_pose_list.append(pose_1)
        
        pose_2 = Pose()
        pose_2.position.x = -0.5
        pose_2.position.y = 5.0
        pose_2.position.z = 0.01
        pose_2.orientation.x = 0.0
        pose_2.orientation.y = 0.0
        pose_2.orientation.z = -0.7071 
        pose_2.orientation.w = 0.7071 
        start_pose_list.append(pose_2)
        
        pose_3 = Pose()
        pose_3.position.x = -1.0
        pose_3.position.y = 4.0
        pose_3.position.z = 0.01
        pose_3.orientation.x = 0.0
        pose_3.orientation.y = 0.0
        pose_3.orientation.z = 0.7071 
        pose_3.orientation.w = 0.7071 
        start_pose_list.append(pose_3)
        
        pose_4 = Pose()
        pose_4.position.x = -1.5
        pose_4.position.y = -1.5
        pose_4.position.z = 0.01
        pose_4.orientation.x = 0.0
        pose_4.orientation.y = 0.0
        pose_4.orientation.z = 0.7071 
        pose_4.orientation.w = 0.7071 
        start_pose_list.append(pose_4)
        
        pose_5 = Pose()
        pose_5.position.x = 2.0
        pose_5.position.y = -1.8
        pose_5.position.z = 0.01
        pose_5.orientation.x = 0.0
        pose_5.orientation.y = 0.0
        pose_5.orientation.z = 0.7071 
        pose_5.orientation.w = 0.7071 
        start_pose_list.append(pose_5)
        
        pose_6 = Pose()
        pose_6.position.x = 6.0
        pose_6.position.y = 7.0
        pose_6.position.z = 0.01
        pose_6.orientation.x = 0.0
        pose_6.orientation.y = 0.0
        pose_6.orientation.z = 0.0
        pose_6.orientation.w = 1.0
        start_pose_list.append(pose_6)
        
        pose_7 = Pose()
        pose_7.position.x = 2.0 
        pose_7.position.y = 7.5
        pose_7.position.z = 0.01
        pose_7.orientation.x = 0.0
        pose_7.orientation.y = 0.0
        pose_7.orientation.z = 1.0
        pose_7.orientation.w = 0.0
        start_pose_list.append(pose_7)
        
        pose_8 = Pose()
        pose_8.position.x = 1.0
        pose_8.position.y = -0.5
        pose_8.position.z = 0.01
        pose_8.orientation.x = 0.0
        pose_8.orientation.y = 0.0
        pose_8.orientation.z = 1.0
        pose_8.orientation.w = 0.0
        start_pose_list.append(pose_8)
        
        pose_9 = Pose()
        pose_9.position.x =  5.0
        pose_9.position.y = -1.5
        pose_9.position.z = 0.01
        pose_9.orientation.x = 0.0
        pose_9.orientation.y = 0.0
        pose_9.orientation.z = 0.0
        pose_9.orientation.w = 1.0
        start_pose_list.append(pose_9)
        
        pose_10 = Pose()
        pose_10.position.x = 2.0
        pose_10.position.y = -0.5
        pose_10.position.z = 0.01
        pose_10.orientation.x = 0.0
        pose_10.orientation.y = 0.0
        pose_10.orientation.z = -0.7071 
        pose_10.orientation.w = 0.7071 
        start_pose_list.append(pose_10)
        
        
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('/unpause simulation service not available, waiting again...')

        unpause = self.unpause.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self.node, unpause)
        
        self.iterator = 0
        index = random.randrange(0, 10, 1)
        req_spawn = SpawnEntity.Request()
        req_spawn.name = 'Turtlebot3'
        req_spawn.xml = self.xml
        req_spawn.robot_namespace = ''
        req_spawn.initial_pose = start_pose_list[index]
        req_spawn.reference_frame = 'world'

        if self.reset_jnts is True:
            # reset simulation
            while not self.delete_entity.wait_for_service(timeout_sec=1.0):
                self.node.get_logger().info('/reset_simulation service not available, waiting again...')
            req_delete = DeleteEntity.Request()
            req_delete.name = 'Turtlebot3'
            
            delete_future = self.delete_entity.call_async(req_delete)
            rclpy.spin_until_future_complete(self.node, delete_future)
            respone = delete_future.result()
            print (respone.success)
            if respone.success == True:
                '''
                while not self.reset_sim.wait_for_service(timeout_sec=1.0):
                    self.node.get_logger().info('/reset_simulation service not available, waiting again...')

                reset_future = self.reset_sim.call_async(Empty.Request())
                rclpy.spin_until_future_complete(self.node, reset_future)
                '''
            
                while not self.add_entity.wait_for_service(timeout_sec=1.0):
                    self.node.get_logger().info('service not available, waiting again...')
                
                spawn_future = self.add_entity.call_async(req_spawn)
                rclpy.spin_until_future_complete(self.node, spawn_future)
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

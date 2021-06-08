""" NOT A REAL TRAINING SCRIPT
Please check the README.md in located in this same folder
for an explanation of this script"""

import gym
import gym_gazebo2
import time
env = gym.make('Turtlebot3-v0')


while True:
    # take a random action
    observation, reward, done, info = env.step(env.action_space.sample())

# -*- coding: utf-8 -*-

"""

"""

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'


import pandapower.networks as pn
import pandapower as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandapower.plotting.plotly import simple_plotly
from pandapower.networks import mv_oberrhein
from pandapower import runpp

import gym
from gym import spaces
from gym.utils import seeding

def main():
    pass


class Environment:
    """
    environment class used for interacting with the power system. The methods
    is copied from gym's environment class.
    """

    reward_range = (-np.inf, np.inf)
    action_space = None
    observation_space = None


    def __init__(self):
        self.state = None
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self,action):
        """
        Takes an action, evaluates the reward
        :param action:
        :return: observation, reward, done, info
        """



    def reset(self):
        """

        :return:observation
        """
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()


class ActionSpace:

    def __init__(self):
        pass


    def sample(self):
        """

        :return: action
        """
        pass





if __name__ == '__main__':
    from optparse import OptionParser
    import inspect







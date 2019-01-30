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


def main():
    pass


class Environment:
    """
    environment class used for interacting with the power system. The methods
    is copied from gym's environment class.
    """

    def __init__(self):
        self.action_space = None #should be a class




    def reset(self):
        pass


    def step(self,action):
        """

        :param action:
        :return: observation, reward, done, info
        """
        pass



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







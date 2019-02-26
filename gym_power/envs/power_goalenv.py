# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import pandas
from gym import GoalEnv, spaces
from gym_power.envs import PowerEnv
"""

"""

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'


class PowerGoalEnv(GoalEnv,PowerEnv):
    def __init__(self):
        super(PowerGoalEnv, self).__init__()
        self.epsilon = 0.3
        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
            observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
        ))


    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Sparse reward setting. 0 if |achieved - desired| < epsilon, -1 else

        :param achieved_goal: achieved goal state
        :type achieved_goal: numpy array
        :param desired_goal: desired goal state
        :type desired_goal: numpy array
        :param info: Can include information for computing the reward
        :return: Reward
        """
        return - (np.linalg.norm(achieved_goal - desired_goal) > self.epsilon)

    def reset(self):
        self._create_initial_state()
        return self._observation()


    def _observation(self):
        pass
if __name__ == '__main__':
    env = PowerGoalEnv()
    print('stopp ikke mobb')
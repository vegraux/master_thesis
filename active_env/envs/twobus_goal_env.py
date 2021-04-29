# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import pandas
from gym import GoalEnv, spaces
from active_env.envs import TwoBusEnv
"""

"""

__author__ = 'Vegard Solberg'
__email__ = 'vegardsolberg@hotmail.com'


class TwoBusGoalEnv(GoalEnv, TwoBusEnv):
    def __init__(self):
        super(TwoBusGoalEnv, self).__init__()
        self.epsilon = 100
        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
            observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
        ))

    def step(self,action):
        episode_over = self._take_action(action)
        if not episode_over:
            obs = self._get_obs()
            info = {"is_success": self._is_success(obs['achieved_goal'],
                                                   obs['desired_goal'])}

            reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        else:
            reward = -1
            obs = None
        return obs, reward, episode_over, {}


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
        if achieved_goal.shape[0] == 1:
            return -1 * (np.linalg.norm(achieved_goal - desired_goal) > self.epsilon)
        else:
            return -1*(np.linalg.norm(achieved_goal - desired_goal,axis=1) > self.epsilon)

    def reset(self):
        self._create_initial_state()
        return self._get_obs()


    def _get_achieved_goal(self):
        return self.powergrid.res_bus.iloc[1, 2]

    def _is_success(self, achieved_goal, desired_goal):
        return self.compute_reward(achieved_goal, desired_goal, None) == 0

    def _get_obs(self):
        obs = self.powergrid.res_bus.values.flatten()
        achieved_goal = self._get_achieved_goal()
        achieved_goal = np.array([achieved_goal])
        obs = {
            "observation": obs,
            "achieved_goal":achieved_goal,
            "desired_goal": np.array([self.desired_goal])
        }
        return obs


if __name__ == '__main__':
    env = TwoBusGoalEnv()
    env.step([-8238])
    print('stopp ikke mobb')
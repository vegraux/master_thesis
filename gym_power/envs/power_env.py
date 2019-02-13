# -*- coding: utf-8 -*-

"""

"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_power.sample_net import simple_two_bus
import pandapower as pp
from pandapower import ppException
import pandas as pd

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'


class PowerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.env = simple_two_bus()
        self.observation_size = 4 * len(
            self.env.bus)  # P,Q,U, delta at each bus
        self.max_power = 4000
        high = np.array([1000000 for _ in range(self.observation_size)])

        self.observation_space = spaces.Box(low=-high, high=high,
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=-self.max_power, high=self.max_power,
                                       shape=(1,), dtype=np.float32)

        self.target_load = 1300

    def step(self, action):
        episode_over = self._take_action(action)
        if not episode_over:
            reward = self._get_reward()
            ob = self._get_obs()

        else:
            reward = -100
            ob = self.reset()

        return ob, reward, episode_over, {}

    def _take_action(self, action):
        """ Converts the action space into an pandapowe action. """
        at = pd.DataFrame(data=action, columns=['p_kw'])
        self.env.gen[at.columns] = at
        try:
            pp.runpp(self.env)
            return False

        except ppException:
            return True

    def _get_reward(self):
        """ Reward is given for scoring a goal."""

        return self.calc_reward()
        # TODO: generalize for different network

    def _get_obs(self):

        return self.env.res_bus.values.flatten()

    def reset(self):
        self.env = simple_two_bus()
        return self._get_obs()

    def calc_reward(self):
        flows = self.env.res_bus
        reward = -np.abs(flows.iloc[1, 2]/self.target_load - 1) #+ \
                 #-np.abs(flows['p_kw'].sum())/self.target_load

        return reward
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        pass


if __name__ == '__main__':
    env = PowerEnv()
    env._step([-1100])
    print(env.metadata)

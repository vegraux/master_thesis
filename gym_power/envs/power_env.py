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
import copy

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'


class PowerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.base_env = simple_two_bus()
        self.env = copy.deepcopy(self.base_env)
        self.observation_size = 4 * len(
            self.env.bus)  # P,Q,U, delta at each bus
        self.max_power = 15000
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
            reward = -20000
            #ob = self.reset()
            ob = None
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
        self.env = copy.deepcopy(self.base_env)
        return self._get_obs()

    def calc_reward(self):
        flows = self.env.res_bus
        reward = -np.abs(flows.iloc[1, 2] -self.target_load) #+ \
                 #-np.abs(flows['p_kw'].sum())/self.target_load

        return reward
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        pass


class PowerEnvSparse(PowerEnv):
    """
    sparse rewards, with desired state in observation
    """
    def __init__(self):

        super(PowerEnvSparse, self).__init__()
        self.observation_size = 4 * len(
            self.env.bus) + 1

        high = np.array([1000000 for _ in range(self.observation_size)])

        self.observation_space = spaces.Box(low=-high, high=high,
                                            dtype=np.float32)

    def calc_reward(self):
        flows = self.env.res_bus
        reward = np.abs(flows.iloc[1, 2] - self.target_load) < 100  # + \
        return reward

    def step(self, action):
        episode_over = self._take_action(action)
        if not episode_over:
            reward = self._get_reward()
            ob = self._get_obs()

        else:
            reward = 0
            #ob = self.reset()
            ob = None
        return ob, reward, episode_over, {}

    def _get_obs(self):
        obs = list(self.env.res_bus.values.flatten())
        obs.append(self.target_load)
        return np.array(obs)


if __name__ == '__main__':
    env = PowerEnv()
    envsparse = PowerEnvSparse()
    envsparse._get_obs()
    env._step([-1100])
    print(env.metadata)

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
        self.seed()
        self.base_powergrid = simple_two_bus()
        self.voltage_threshold = 0.05
        self.power_threshold = 0.05
        self.nominal_cos_phi = 0.8
        self.cos_phi_threshold = 0.1
        self.powergrid = copy.deepcopy(self.base_powergrid)
        self.observation_size = 4 * len(
            self.powergrid.bus)  # P,Q,U, delta at each bus
        self.max_power = 30000
        high = np.array([1000000 for _ in range(self.observation_size)])

        self.observation_space = spaces.Box(low=-high, high=high,
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=-self.max_power, high=self.max_power,
                                       shape=(1,), dtype=np.float32)

        self.desired_goal = 8000

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
        self.powergrid.gen[at.columns] = at
        try:
            pp.runpp(self.powergrid)
            return False

        except ppException:
            return True

    def _get_reward(self):
        """ Reward is given for scoring a goal."""

        return self.calc_reward()
        # TODO: generalize for different network

    def _get_obs(self):
        """
        samples an initial state for the power system
        :return:
        """
        return self.powergrid.res_bus.values.flatten()

    def reset(self):
        self.powergrid = copy.deepcopy(self.base_powergrid)
        self._create_initial_state()

        return self._get_obs()


    def _create_initial_state(self):
        nominal_voltage = self.powergrid.res_bus['vm_pu'] #Not able to set start voltage at the buses
        start_voltages = self._random_uniform(nominal_voltage,self.voltage_threshold)

        nominal_power = self.powergrid.gen['sn_kva']
        nominal_cos_phi = np.ones_like(nominal_power) * self.nominal_cos_phi
        cos_phi = self._random_uniform(nominal_cos_phi,self.cos_phi_threshold)
        start_power = nominal_power*cos_phi
        start_power = pd.Series(start_power,name='p_kw')
        self.powergrid.gen['p_kw'] = -start_power







    def _random_uniform(self,nominal_values,threshold):
        """
        Draws random values uniformly around nominal values
        :param nominal_values: nominal values for voltage/angles
        :param threshold: Usually 0.05, as is normal in the power grid.
        :return: start_values
        """
        nr = len(nominal_values)
        return nominal_values*((self.np_random.rand(nr)*2 - 1)*threshold + 1)


    def calc_reward(self):
        flows = self.powergrid.res_bus
        reward = -(flows.iloc[1, 2] - self.desired_goal) ** 2 #+ \
                 #-np.abs(flows['p_kw'].sum())/self.desired_goal

        return reward
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        pass



if __name__ == '__main__':
    env = PowerEnv()
    env.step([-1200])
    print(env.metadata)

# -*- coding: utf-8 -*-

"""

"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from active_env.sample_net import simple_two_bus
import pandapower as pp
from pandapower import ppException
import pandas as pd

import copy

__author__ = 'Vegard Solberg'
__email__ = 'vegardsolberg@hotmail.com'


class TwoBusEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,episode_length=200):
        self.seed()
        self.base_powergrid = simple_two_bus()
        self.voltage_threshold = 0.05
        self.power_threshold = 0.05
        self.nominal_cos_phi = 0.8
        self.cos_phi_threshold = 0.1
        self.powergrid = copy.deepcopy(self.base_powergrid)
        self.observation_size = 4 * len(
            self.powergrid.bus)  # P,Q,U, delta at each bus
        self.max_power = 20
        high = np.array([1000000 for _ in range(self.observation_size)])

        self.observation_space = spaces.Box(low=-high, high=high,
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(1,), dtype=np.float32)

        self.desired_goal = 1300
        self.episode_length = episode_length
        self.current_step = 0

    def step(self, action):
        #scaled_action = 0.5*(action + 1)*self.max_power
        scaled_action = action*self.max_power
        episode_over = self._take_action(scaled_action)
        self.current_step += 1
        if self.current_step > self.episode_length:
            ob = self.reset()
            self.current_step = 0

        if not episode_over:
            reward = self._get_reward()
            ob = self._get_obs()

        else:
            reward = -20000
            ob = self.reset()
        self.current_step += 1

        return ob, reward, episode_over, {}

    def _take_action(self, action):
        """ Converts the action space into an pandapowe action. """
        at = pd.DataFrame(data=[action], columns=['p_kw'])
        #self.powergrid.gen[at.columns] = at
        self.powergrid.gen[at.columns] += at
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
        reward = -np.abs(flows.iloc[1, 2] - self.desired_goal)

        return reward
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        pass

class PowerEnvOld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.powergrid = simple_two_bus()
        self.observation_size = 4 * len(
            self.powergrid.bus)  # P,Q,U, delta at each bus
        self.max_power = 2000
        high = np.array([1000000 for _ in range(self.observation_size)])

        self.observation_space = spaces.Box(low=-high, high=high,
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=0, high=self.max_power,
                                       shape=(1,), dtype=np.float32)

        self.target_load = 1300



    def step(self, action):
        episode_over = self._take_action(action)
        if episode_over:
            reward = -20000
            ob = self.reset()

        else:
            reward = self._get_reward()
            ob = self._get_obs()
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
        reward = -np.abs(self.powergrid.res_bus.iloc[1, 2] - self.target_load)
        return reward
        # TODO: generalize for different network

    def _get_obs(self):
        return self.powergrid.res_bus.values.flatten()

    def reset(self):
        self.powergrid = simple_two_bus()
        return self._get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        pass


class PowerEnvOldNormalized(PowerEnvOld):
    def __init__(self):
        super(PowerEnvOldNormalized, self).__init__()
        self.base_power = self.powergrid.gen['sn_kva'].max()

    def _get_obs(self):
        state = copy.copy(self.powergrid.res_bus)
        state['p_kw'] /= self.base_power
        state['q_kvar'] /= self.base_power
        return state.values.flatten()


    def step(self, action):
        scaled_action = action*self.max_power
        episode_over = self._take_action(scaled_action)
        if episode_over:
            reward = -10
            ob = self.reset()

        else:
            reward = self._get_reward()
            ob = self._get_obs()
        return ob, reward, episode_over, {}

    def _get_reward(self):
        reward = -np.abs(self.powergrid.res_bus.iloc[1, 2] - self.target_load)/self.target_load
        return reward

if __name__ == '__main__':
    env = TwoBusEnv()
    obs, rewards, done, info = env.step([-1200])
    print(env.metadata)

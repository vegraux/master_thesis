# -*- coding: utf-8 -*-

"""
environment for active network management for controlling flexible load
and/or generation control
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_power.sample_net import simple_two_bus
from pandapower.networks import mv_oberrhein
import copy
import pandapower as pp
from pandapower import ppException
import pandas as pd
import enlopy as el



__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'

class ActiveEnv(gym.Env):

    def __init__(self,episode_length=200):
        obs = self.reset()
        self.seed()
        self.base_powergrid = mv_oberrhein()
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

        self.episode_length = episode_length
        self.current_step = 0
        self.load_forcasts = self.get_initial_load_forecast()
        self.load_dict = self.get_load_dict()
        self.flexible_load_indices = np.random.choice(len(self.load_forcasts),10)

    def get_load_dict(self):
        """
        :return: dictionary mapping columns name to index
        """
        return {col: index for (index, col) in enumerate(self.powergrid.load)}


    def get_forecast(self):
        """
        Finds the forecasted hourly demand for the next 24 hours
        :return: dictionary with
        """
        forecasts = []
        t = self.current_step
        for load in self.load_forcasts:
            day_forecast = load[t:t+24]
            forecasts.append(day_forecast)

        return forecasts


    def get_initial_load_forecast(self):
        """
        gets the forecasts for all loads
        :return:
        """
        nr_days = (self.episode_length // 24) + 2
        initial_forecasts = []
        for k in range(len(self.powergrid.load)):
            load_forcast = []
            for day in range(nr_days):
                day = list(el.generate.gen_daily_stoch_el())
                load_forcast += day

            load_forcast = np.array(load_forcast)
            initial_forecasts.append(load_forcast)

        return initial_forecasts

    def _create_initial_state(self):
        """
        Creates the initial network using the forecast
        :return:
        """
        loads = []
        for load in self.load_forcasts:
            loads.append(load[0])

        self.powergrid.load['p_kw'] = loads

    def reset(self):
        self.powergrid = copy.deepcopy(self.base_powergrid)
        self._create_initial_state()

        return self._get_obs()

    def get_bus_state(self):
        """
        Return the voltage, active and reactive power at every bus
        :return:
        """
        return self.powergrid.res_bus.values.flatten()



    def _get_obs(self):
        """
        samples an initial state for the power system
        :return:
        """
        forecasts = self.get_forecast()
        bus_state = self.get_bus_state()

        pass

    def take_action(self, action):
        """
        take the action vector and modifies the flexible loads
        :return:
        """
        load_index = self.load_dict['p_kw']
        self.powergrid.load.iloc[self.flexible_load_indices, load_index] = action
        try:
            pp.runpp(self.powergrid)
            return False

        except ppException:
            return True



    def step(self, action):

        episode_over = self._take_action(action)
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

        return ob, reward, episode_over, {}












if __name__ == '__main__':

    env = ActiveEnv()
    print('her ska det stopeps')
    env.get_forecast()





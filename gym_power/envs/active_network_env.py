# -*- coding: utf-8 -*-

"""
environment for active network management for controlling flexible load
and/or generation control
"""
import os
import dotenv
dotenv.load_dotenv()
import gym
import matplotlib.pyplot as plt
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_power.sample_net import simple_two_bus
from pandapower.networks import create_cigre_network_mv
import copy
import pandapower as pp
from pandapower import ppException
import pandas as pd
import enlopy as el



__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'
DATA_PATH = os.getenv('DATA_PATH')

class ActiveEnv(gym.Env):

    def __init__(self,episode_length=200, mod_period=2):
        self.np_random = None
        self.seed()
        #time attributes
        self.current_step = 0
        self.episode_start_hour = self.select_start_hour()
        self.episode_length = episode_length

        #power grid
        self.base_powergrid = create_cigre_network_mv(with_der="pv_wind")
        self.powergrid = copy.deepcopy(self.base_powergrid)
        pp.runpp(self.powergrid)
        self.flexible_load_indices = np.arange(len(self.powergrid.load))
        self.max_power = 20
        self.last_action = np.zeros_like(self.flexible_load_indices)

        #state variables, forecast + commitment
        self.solar_data = self.load_solar_data()
        self.solar_forecasts = self.get_episode_solar_forecast()
        self.demand_forcasts = self.get_episode_demand_forecast()
        self.mod_period = mod_period
        self._commitments = np.zeros(len(self.powergrid.load)) != 0



        self.observation_size = self._get_obs().shape[0]
        high = np.array([np.inf for _ in range(self.observation_size)])

        self.observation_space = spaces.Box(low=-high, high=high,
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(1,), dtype=np.float32)


        self.load_dict = self.get_load_dict()





    def get_load_dict(self):
        """
        :return: dictionary mapping columns name to index
        """
        return {col: index for (index, col) in enumerate(self.powergrid.load)}


    def get_demand_forecast(self):
        """
        Finds the forecasted hourly demand for the next 24 hours
        :return: List with 24 demand for all buses
        """
        forecasts = []
        t = self.current_step
        for load in self.demand_forcasts:
            day_forecast = load[t:t+24]
            forecasts.append(day_forecast)

        return forecasts
    def get_solar_forecast(self):
        """
        Returns solar forecast for the next 24 hours.
        :return:
        """
        t = self.current_step
        return self.solar_forecasts[t:t+24]

    def get_episode_demand_forecast(self,scale_demand=30000):
        """
        gets the forecasts for all loads in the episode
        :return:
        """
        nr_days = (self.episode_length // 24) + 3 #margin
        episode_forecasts = []
        for k in range(len(self.powergrid.load)):
            demand_forcast = []
            for day in range(nr_days):
                day = list(el.generate.gen_daily_stoch_el())
                demand_forcast += day

            demand_forcast = scale_demand*np.array(demand_forcast)
            episode_forecasts.append(demand_forcast[self.episode_start_hour:])

        return episode_forecasts

    def _create_initial_state(self):
        """
        Creates the initial network using the forecast
        :return:
        """
        loads = []
        for load in self.demand_forcasts:
            loads.append(load[0])

        self.powergrid.load['p_kw'] = loads

    def select_start_hour(self):
        """
        Selects start hour for the episode
        :return:
        """
        return self.np_random.choice(24)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        assert self.current_step == 0

        self.powergrid = copy.deepcopy(self.base_powergrid)
        self._create_initial_state()
        self.episode_start_hour = self.select_start_hour()
        self.solar_forecasts = self.get_episode_solar_forecast()
        self.demand_forcasts = self.get_episode_demand_forecast()

        return self._get_obs()

    def get_bus_state(self):
        """
        Return the voltage, active and reactive power at every bus
        :return:
        """
        return self.powergrid.res_bus.values.flatten()


    def get_episode_solar_forecast(self):
        """
        Method that returns the solar_forcast for the entire episode
        """
        start_day = self.np_random.choice(350) #margin for episode length
        start = self.episode_start_hour + start_day*24
        nr_hours = self.episode_length + 25 #margin of 1
        episode_solar_forecast = self.solar_data[start:start+nr_hours].values
        return episode_solar_forecast.ravel()

    def get_commitment_state(self):
        """
        Transforms _commitments array from booleans to 0,1
        :return:
        """
        commitments = np.zeros(self._commitments.shape)
        commitments[self._commitments] = 1
        return commitments


    def load_solar_data(self):
        solar_path = os.path.join(DATA_PATH,'solar_irradiance_2015.hdf')
        return pd.read_hdf(solar_path,key='sun')

    def _check_commitment(self, action):
        """
        Checks if a load has a commitment in terms of production due
        to its action from last step, and modifies the current action to the
        opposite of the last action, so the consumption is not altered.
        """

        action[self._commitments] = - self.last_action[self._commitments]

        new_commitments = (action != 0)
        new_commitments[self._commitments] = False
        self._commitments = new_commitments
        return action



    def _get_obs(self):
        """
        samples an state for the power system
        :return:
        """
        demand_forecasts = self.get_demand_forecast()
        solar_forecasts = self.get_solar_forecast()
        bus_state = self.get_bus_state()
        commitment_state = self.get_commitment_state()

        state = []
        for demand in demand_forecasts:
            state += list(demand)

        state += list(solar_forecasts)
        state += list(bus_state)
        state += list(commitment_state)

        return np.array(state)

    def _take_action(self, action):
        """
        take the action vector and modifies the flexible loads
        :return:
        """
        action = self._check_commitment(action)
        load_index = self.load_dict['p_kw']

        self.powergrid.load.iloc[self.flexible_load_indices, load_index] = action
        try:
            pp.runpp(self.powergrid)
            return False

        except ppException:
            return True


    def _get_reward(self):
        pass




    def step(self, action):
        episode_over = self._take_action(action)
        self.current_step += 1
        if self.current_step > self.episode_length:
            self.current_step = 0
            ob = self.reset()

        if not episode_over:
            reward = self._get_reward()
            ob = self._get_obs()

        else:
            reward = -20000
            ob = self.reset()

        return ob, reward, episode_over, {}












if __name__ == '__main__':
    env = ActiveEnv()
    demand = env.get_demand_forecast()
    solar = env.get_solar_forecast()
    action = np.ones_like(env.last_action)
    ob, reward, episode_over, info = env.step(action)
    print('para aquí')

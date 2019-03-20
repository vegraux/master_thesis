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
from gym_power.sample_net import simple_two_bus, cigre_network
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

    def __init__(self,episode_length=200, look_ahead=4, solar_scale=0.8,
                 do_action=True):
        self.np_random = None
        self.seed()
        #time attributes
        self.current_step = 0
        self._episode_start_hour = self.select_start_hour()
        self.episode_length = episode_length
        self.look_ahead = look_ahead
        self.do_action = do_action

        #power grid
        self.base_powergrid = cigre_network()
        pp.runpp(self.base_powergrid)
        self.powergrid = copy.deepcopy(self.base_powergrid)
        self.flexible_load_indices = np.arange(len(self.powergrid.load))
        self.last_action = np.zeros_like(self.flexible_load_indices)
        self.pq_ratio = self.calc_pq_ratio()

        #state variables, forecast + commitment
        self.solar_data = self.load_solar_data(solar_scale=solar_scale)
        self.solar_forecasts = self.get_episode_solar_forecast()
        self.demand_forcasts = self.get_episode_demand_forecast()
        self.set_demand_and_solar()
        self._commitments = np.zeros(len(self.powergrid.load)) != 0


        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=self._get_obs().shape,
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., shape=self.last_action.shape,
                                       dtype=np.float32)


        self.load_dict = self.get_load_dict()





    def get_load_dict(self):
        """
        :return: dictionary mapping columns name to index
        """
        return {col: index for (index, col) in enumerate(self.powergrid.load)}

    def calc_pq_ratio(self):
        """
        Power factor for loads are constant. This method finds the PQ-ratio
        for all loads (same as for default cigre network)
        """
        net = create_cigre_network_mv(with_der="pv_wind")
        pq_ratio = net.load['q_kvar'] / net.load['p_kw']
        return pq_ratio

    def get_demand_forecast(self):
        """
        Finds the forecasted hourly demand for the next T hours
        :return: List with T demand for all buses
        """
        forecasts = []
        t = self.current_step
        for load in self.demand_forcasts:
            day_forecast = load[t:t+self.look_ahead]
            forecasts.append(day_forecast)

        return forecasts




    def get_solar_forecast(self):
        """
        Returns solar forecast for the next look_ahead hours.
        :return:
        """
        t = self.current_step
        return self.solar_forecasts[t:t+self.look_ahead]

    def get_scaled_solar_forecast(self):
        """
        scales each solar panel production with nominal values.
        :return: Sum of solar production in network (kW)
        """
        solar_pu = self.solar_forecasts
        scaled_solar = []
        for sol in solar_pu:
            scaled_solar.append((sol*self.powergrid.sgen['sn_kva']).sum())
        return np.array(scaled_solar)

    def get_scaled_demand_forecast(self):
        demand_pu = self.demand_forcasts[0]
        scaled_demand = []
        for demand in demand_pu:
            scaled_demand.append((demand*self.powergrid.load['sn_kva']).sum())
        return np.array(scaled_demand)



    def get_episode_demand_forecast(self,scale_demand=10):
        """
        gets the forecasts for all loads in the episode
        :return:
        """
        nr_days = (self.episode_length // 24) + 3 #margin
        episode_forecasts = []
        for k in range(1):#range(len(self.powergrid.load)):
            demand_forcast = []
            for day in range(nr_days):
                day = list(el.generate.gen_daily_stoch_el())
                demand_forcast += day

            demand_forcast = scale_demand*np.array(demand_forcast)
            episode_forecasts.append(demand_forcast[self._episode_start_hour:])

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
        self.current_step = 0

        self.powergrid = copy.deepcopy(self.base_powergrid)
        self._episode_start_hour = self.select_start_hour()
        self.solar_forecasts = self.get_episode_solar_forecast()
        self.demand_forcasts = self.get_episode_demand_forecast()
        self.set_demand_and_solar()

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
        start = self._episode_start_hour + start_day * 24
        nr_hours = self.episode_length + self.look_ahead + 1 #margin of 1
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


    def load_solar_data(self,solar_scale=0.5):
        solar_path = os.path.join(DATA_PATH,'solar_irradiance_2015.csv')
        solar = pd.read_csv(solar_path)
        solar.index = pd.to_datetime(solar.iloc[:, 0])
        solar.index.name = 'time'
        solar = solar.iloc[:, [1]]
        return solar_scale*solar

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
        self.last_action = action
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

    def _take_action(self, action, flexibility=0.1):
        """
        Takes the action vector, Scales it and modifies the flexible loads
        :return:
        """
        action *= flexibility*self.powergrid.load['p_kw']
        action = self._check_commitment(action)
        load_index = self.load_dict['p_kw']

        self.powergrid.load.iloc[self.flexible_load_indices, load_index] += action
        try:
            pp.runpp(self.powergrid)
            return False

        except ppException:
            return True

    def set_demand_and_solar(self):
        """
        Updates the demand and solar production according to the forecasts
        """
        solar_pu = self.get_solar_forecast()[0]
        demand_pu = self.get_demand_forecast()[0][0]
        self.powergrid.sgen['p_kw'] = - solar_pu * self.powergrid.sgen['sn_kva']
        self.powergrid.load['p_kw'] = demand_pu * self.powergrid.load['sn_kva']
        self.powergrid.load['q_kvar'] = self.powergrid.load['p_kw'] * self.pq_ratio



    def _get_reward(self,v_min=0.95,v_max=1.05, i_max_loading= 90):
        v = self.powergrid.res_bus['vm_pu']
        v_lower = sum(v_min - v[v < v_min])
        v_over =  sum(v[v > v_max] - v_max)

        i = self.powergrid.res_line['loading_percent']
        i_over = sum(i[i > i_max_loading] - i_max_loading)

        i_loss = sum(self.powergrid.res_line['pl_kw'])

        state_loss = v_lower + v_over + i_over + i_loss
        return -state_loss






    def step(self, action):
        self.set_demand_and_solar()

        if self.do_action:
            episode_over = self._take_action(action)
        else:
            pp.runpp(self.powergrid)
            episode_over = False

        self.current_step += 1
        if self.current_step > self.episode_length:
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
    hours = 100
    load = env.get_scaled_demand_forecast()
    sol = env.get_scaled_solar_forecast()
    fig, ax = plt.subplots()

    plt.plot(sol[:hours],axes=ax)
    plt.plot(load[:hours],axes=ax)
    plt.show()
    for hour in range(hours):
        action = env.action_space.sample()
        ob, reward, episode_over, info = env.step(action)
        print(reward,load[hour],sol[hour])
    print('para aquí')

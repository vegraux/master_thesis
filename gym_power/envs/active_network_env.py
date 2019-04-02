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

    params = {'episode_length': 200,
              'reward_terms':['voltage','current','imbalance','activation'],
              'voltage_weight':1,
              'current_weight':0.01,
              'imbalance_weight':1e-4,
              'activation_weight':1e-4,
              'forecast_horizon':4,
              'flexibility':0.1,
              'solar_scale':0.8,
              'demand_scale':10,
              'state_space': ['sun','demand','bus','imbalance'],
              'v_upper':1.04,
              'v_lower':0.96,
              'i_upper':90,
              }

    def set_parameters(self, new_parameters):
        """
        sets params for animals
        :param new_parameters: New parameter value
        :type new_parameters: dictionary
        """
        allowed_keys = ['episode_length', 'reward_terms', 'voltage_weight',
                        'current_weight', 'imbalance_weight',
                        'forecast_horizon', 'activation_weight', 'flexibility',
                        'state_space','solar_scale', 'demand_scale']
        non_negative = ['voltage_weight', 'current_weight', 'imbalance_weight',
                        'activation_weight']
        zero_to_one = ['flexibility']
        for key in new_parameters:
            if key not in allowed_keys:
                raise KeyError('Invalid parameter name' + key)
            if key in non_negative and new_parameters[key] < 0:
                raise ValueError('Invalid parameter value, negative values '
                                 'not allowed: ' + key)
            if key in zero_to_one and (
                    new_parameters[key] < 0 or new_parameters[key] > 1):
                raise ValueError('Invalid parameter value, value must be'
                                 ' between 0 and 1: ' + key)

            self.params = {**self.params, **new_parameters}

    def __init__(self,episode_length=200, look_ahead=4,
                 do_action=True, flexibility=0.1, force_commitments=False,
                 bus_in_state=False):
        self.np_random = None
        self.seed()
        #time attributes
        self._current_step = 0
        self._episode_start_hour = self.select_start_hour()
        self.do_action = do_action

        #power grid
        self.base_powergrid = cigre_network()
        pp.runpp(self.base_powergrid)
        self.powergrid = copy.deepcopy(self.base_powergrid)
        self.flexible_load_indices = np.arange(len(self.powergrid.load))
        self.last_action = np.zeros_like(self.flexible_load_indices)
        self.pq_ratio = self.calc_pq_ratio()


        #state variables, forecast + commitment
        self.solar_data = self.load_solar_data()
        self.solar_forecasts = self.get_episode_solar_forecast()
        self.demand_forcasts = self.get_episode_demand_forecast()
        self.set_demand_and_solar()
        self.force_commitments = force_commitments
        self._commitments = np.zeros(len(self.powergrid.load)) != 0
        self.resulting_demand = np.zeros(self.params['episode_length'])
        self._imbalance = np.zeros(self.params['episode_length'])



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
        t = self._current_step
        horizon = self.params['forecast_horizon']
        for load in self.demand_forcasts:
            day_forecast = load[t:t+horizon]
            forecasts.append(day_forecast)

        return forecasts




    def get_solar_forecast(self):
        """
        Returns solar forecast for the next look_ahead hours.
        :return:
        """
        t = self._current_step
        horizon = self.params['forecast_horizon']
        return self.solar_forecasts[t:t+horizon]

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



    def get_episode_demand_forecast(self):
        """
        gets the forecasts for all loads in the episode
        :return:
        """
        episode_length = self.params['episode_length']
        demand_scale = self.params['demand_scale']
        nr_days = (episode_length // 24) + 3 #margin
        episode_forecasts = []
        for k in range(1):#range(len(self.powergrid.load)):
            demand_forcast = []
            for day in range(nr_days):
                day = list(el.generate.gen_daily_stoch_el())
                demand_forcast += day

            demand_forcast = demand_scale*np.array(demand_forcast)
            episode_forecasts.append(demand_forcast[self._episode_start_hour:])

        return episode_forecasts



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
        self._current_step = 0

        self.powergrid = copy.deepcopy(self.base_powergrid)
        self._episode_start_hour = self.select_start_hour()
        self.solar_forecasts = self.get_episode_solar_forecast()
        self.demand_forcasts = self.get_episode_demand_forecast()
        self.set_demand_and_solar()
        self._imbalance = np.zeros(self.params['episode_length'])

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
        episode_length = self.params['episode_length']
        horizon = self.params['forecast_horizon']
        start_day = self.np_random.choice(365-episode_length//24 - 2)
        start = self._episode_start_hour + start_day * 24
        nr_hours = episode_length + horizon + 1 #margin of 1
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
        solar_path = os.path.join(DATA_PATH,'solar_irradiance_2015.csv')
        solar = pd.read_csv(solar_path)
        solar.index = pd.to_datetime(solar.iloc[:, 0])
        solar.index.name = 'time'
        solar = solar.iloc[:, [1]]

        return self.params['solar_scale'] * solar

    def _check_commitment(self, action):
        """
        Checks if a load has a commitment in terms of production due
        to its action from last step, and modifies the current action to the
        opposite of the last action, so the consumption is not altered.
        """
        if self.force_commitments:
            action[self._commitments] = - self.last_action[self._commitments]

            new_commitments = (action != 0)
            new_commitments[self._commitments] = False
            self._commitments = new_commitments
            self.last_action = action
            return action
        else:
            return action



    def _get_obs(self):
        """
        returns the state for the power system
        :return:
        """
        state = []
        if 'demand' in self.params['state_space']:
            demand_forecasts = self.get_demand_forecast()
            for demand in demand_forecasts:
                state += list(demand)

        if 'sun' in self.params['state_space']:
            solar_forecasts = self.get_solar_forecast()
            state += list(solar_forecasts)

        if 'bus' in self.params['state_space']:
            bus_state = self.get_bus_state()
            state += list(bus_state)

        if self.force_commitments:
            commitment_state = self.get_commitment_state()
            state += list(commitment_state)

        if 'imbalance' in self.params['state_space']:
            balance = self.calc_imbalance() / 30000
            state += [balance]

        return np.array(state)

    def calc_imbalance(self):
        """
        Calculates how much power the agent ows to the system, i.e the amount
        of extra energy the loads have received the last 24 hours. Reward function
        penalises a large balance.
        :return:

        """
        t = self._current_step
        if t > 24:
            modifications = self._imbalance[t - 24:t]

        else:
            modifications = self._imbalance[:t]

        return modifications.sum()


    def log_resulting_demand(self):
        """
        Logs the resulting demand in an episode after the agent has taken
        its actions

        """
        loads = self.powergrid.load['p_kw']
        self.resulting_demand[self._current_step] = loads.sum()


    def _take_action(self, action):
        """
        Takes the action vector, scales it and modifies the flexible loads
        :return:
        """

        action *= self.params['flexibility'] * self.powergrid.load['p_kw']
        if self.force_commitments:
            action = self._check_commitment(action)

        self._imbalance[self._current_step] = action.sum()

        load_index = self.load_dict['p_kw']

        self.powergrid.load.iloc[self.flexible_load_indices, load_index] += action
        try:
            pp.runpp(self.powergrid)
            self.log_resulting_demand()
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



    def calc_reward(self, old_imbalance,action,include_loss=False):

        state_loss = 0
        if 'voltage' in self.params['reward_terms']:
            v = self.powergrid.res_bus['vm_pu']
            v_min = self.params['v_lower']
            v_max = self.params['v_upper']
            v_lower = sum(v_min - v[v < v_min])*self.params['voltage_weight']
            v_over =  sum(v[v > v_max] - v_max)*self.params['voltage_weight']
            state_loss += (v_lower + v_over)

        if 'current' in self.params['reward_terms']:
            i = self.powergrid.res_line['loading_percent']
            i_max = self.params['i_upper']
            i_over = sum(i[i > i_max] - i_max) * self.params['current_weight']
            state_loss += i_over

        if 'imbalance' in self.params['reward_terms']:
            balance = self.calc_imbalance()
            balance_change = np.abs(balance) - np.abs(old_imbalance)
            state_loss += balance_change*self.params['imbalance_weight']

        if 'activation' in self.params['reward_terms']:
            action *= self.params['flexibility'] * self.powergrid.load['p_kw']
            act_loss = np.abs(action).sum()*self.params['activation_weight']
            state_loss +=  act_loss


        if include_loss:
            i_loss = sum(self.powergrid.res_line['pl_kw'])
            state_loss += i_loss

        return -state_loss

    def plot_demand_and_solar(self, hours=100):
        """
        Visualise the total solar production and demand for buses in the system
        :param hours:
        :return:
        """
        load = self.get_scaled_demand_forecast()
        sol = self.get_scaled_solar_forecast()
        resulting_demand = self.resulting_demand
        fig, ax = plt.subplots()
        plt.plot(sol[:hours], axes=ax)
        plt.plot(load[:hours], axes=ax)
        plt.plot(resulting_demand[:hours], axes=ax)
        plt.legend(['solar','demand','modified'])
        plt.show()




    def step(self, action):
        self.set_demand_and_solar()
        old_balance = self.calc_imbalance()
        if self.do_action:
            episode_over = self._take_action(action)
        else:
            pp.runpp(self.powergrid)
            episode_over = False

        self._current_step += 1
        if self._current_step >= self.params['episode_length']:
            ob = self.reset()

        if not episode_over:
            reward = self.calc_reward(old_balance,action)
            ob = self._get_obs()

        else:
            reward = -10
            ob = self.reset()

        return ob, reward, episode_over, {}


    def render(self, mode='human', close=False):
        pass









if __name__ == '__main__':
    env = ActiveEnv()
    env.set_parameters({'activation_weight':1e-4})
    for hour in range(28):
        action = env.action_space.sample()
        ob, reward, episode_over, info = env.step(action)
    solar_forecast = env.get_solar_forecast()
    env.calc_imbalance()

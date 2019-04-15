# -*- coding: utf-8 -*-

"""
environment for active network management for controlling flexible load
and/or generation control
"""
import os
import pickle

import dotenv

from stable_baselines import DDPG

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
              'demand_std':0.03,
              'solar_std':0.03,
              'total_imbalance':False}

    def set_parameters(self, new_parameters):
        """
        sets params for animals
        :param new_parameters: New parameter value
        :type new_parameters: dictionary
        """
        allowed_keys = ['episode_length', 'reward_terms', 'voltage_weight',
                        'current_weight', 'imbalance_weight',
                        'forecast_horizon', 'activation_weight', 'flexibility',
                        'state_space','solar_scale', 'demand_scale','v_lower',
                        'v_upper','i_upper', 'demand_std','solar_std',
                        'total_imbalance']
        non_negative = ['voltage_weight', 'current_weight', 'imbalance_weight',
                        'activation_weight']
        zero_to_one = ['flexibility']
        for key in new_parameters:
            if key not in allowed_keys:
                raise KeyError('Invalid parameter name: ' + key)
            if key in non_negative and new_parameters[key] < 0:
                raise ValueError('Invalid parameter value, negative values '
                                 'not allowed: ' + key)
            if key in zero_to_one and (
                    new_parameters[key] < 0 or new_parameters[key] > 1):
                raise ValueError('Invalid parameter value, value must be'
                                 ' between 0 and 1: ' + key)


        self.params = {**self.params, **new_parameters}

        if ('state_space' in new_parameters) or ('total_imbalance' in new_parameters):
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                shape=self._get_obs().shape,
                                                dtype=np.float32)
        #if ('demand_std' in new_parameters) or ('solar_std' in new_parameters):
        #    self.set_demand_and_solar()

        _ = self.reset()

    def __init__(self,do_action=True, force_commitments=False, seed=None):
        self.np_random = None
        self._seed = self.seed(seed)
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
        self.demand_forecasts = self.get_episode_demand_forecast()
        self.set_demand_and_solar()
        self.force_commitments = force_commitments
        self._commitments = np.zeros(len(self.powergrid.load)) != 0
        self.resulting_demand = np.zeros(self.params['episode_length'])
        self._imbalance = self.empty_imbalance()



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
        for load in self.demand_forecasts:
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
        demand_pu = self.demand_forecasts[0]
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
        self.demand_forecasts = self.get_episode_demand_forecast()
        self.set_demand_and_solar()
        self._imbalance = self.empty_imbalance()

        return self._get_obs()

    def empty_imbalance(self):
        """
        Creates imbalance array, all 0.
        :return:
        """
        nr_loads = len(self.flexible_load_indices)
        episode_length = self.params['episode_length']
        return np.zeros((nr_loads,episode_length))

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
        solar_days = (self.solar_data.index[-1] - self.solar_data.index[0])
        solar_days = solar_days.days
        start_day = self.np_random.choice(solar_days-episode_length//24 - 2)
        start = self._episode_start_hour + start_day * 24
        nr_hours = episode_length + horizon + 1 #margin of 1
        episode_solar_forecast = self.solar_data[start:start+nr_hours].values
        return episode_solar_forecast.ravel() *self.params['solar_scale']

    def get_commitment_state(self):
        """
        Transforms _commitments array from booleans to 0,1
        :return:
        """
        commitments = np.zeros(self._commitments.shape)
        commitments[self._commitments] = 1
        return commitments


    def load_solar_data(self):
        solar_path = os.path.join(DATA_PATH,'hourly_solar_data.csv')
        solar = pd.read_csv(solar_path)
        solar.index = pd.to_datetime(solar.iloc[:, 0])
        solar.index.name = 'time'
        solar = solar.iloc[:, [1]]

        return solar

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
            if self.params['total_imbalance']:
                state += [balance.sum()]
            else:
                state +=list(balance)

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
            modifications = self._imbalance[:,t - 24:t]

        elif t>0:
            modifications = self._imbalance[:,:t]

        else:
            modifications = np.zeros((len(self.flexible_load_indices),1))

        return modifications.sum(axis=1)


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
        nominal_load = self.powergrid.load['sn_kva']
        forecasted_load = nominal_load*self.get_demand_forecast()[0][0]

        action *= self.params['flexibility'] * forecasted_load
        if self.force_commitments:
            action = self._check_commitment(action)

        self._imbalance[:,self._current_step] = action

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
        Updates the demand and solar production according to the forecasts, with
        some n
        """
        solar_pu = self.get_solar_forecast()[0]
        solar_pu += solar_pu*self.params['solar_std']*self.np_random.randn()

        demand_pu = self.get_demand_forecast()[0][0]
        demand_pu += demand_pu*self.params['demand_std']*self.np_random.randn()

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
            state_loss += balance_change.sum()*self.params['imbalance_weight']

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
        reward = self.calc_reward(old_balance, action)

        if (self._current_step >= self.params['episode_length']) or episode_over:
            ob = self.reset()
        else:
            ob = self._get_obs()

        return ob, reward, episode_over, {}


    def render(self, mode='human', close=False):
        pass






def load_env(model_name='flexible_load_first',seed=9):
#flexible_load_first, overnight, larger_margin_cost, discount_06, flex50
    location = 'C:\\Users\\vegar\\Dropbox\\Master\\thesis.git\\RLpower\\models\\'
    params_name = model_name +'_params.p'
    model = DDPG.load(location + model_name)
    env = ActiveEnv(seed=seed)
    with open(location + params_name,'rb') as f:
        params = pickle.load(f)

    env.set_parameters(params)
    model.set_env(env)
    return model, env


if __name__ == '__main__':
    env = ActiveEnv()
    env.set_parameters({'solar_std': 0.1})
    nominal_sun = env.powergrid.sgen['sn_kva']
    solar_forecast = nominal_sun * env.get_solar_forecast()[0]
    solar = -env.powergrid.sgen['p_kw']
    assert np.linalg.norm(solar_forecast - solar) > 0.1
# -*- coding: utf-8 -*-

"""
Gym environment implementing demand response in a pandapower net.
"""
import os
import pickle
import dotenv
from stable_baselines import DDPG

dotenv.load_dotenv(dotenv.find_dotenv())
import gym
import matplotlib.pyplot as plt
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym_power.sample_net import cigre_network
from pandapower.networks import create_cigre_network_mv
import copy
import pandapower as pp
from pandapower import ppException
import pandas as pd

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'
DATA_PATH = os.getenv('DATA_PATH')


class ActiveEnv(gym.Env):
    params = {'episode_length': 200,
              'reward_terms': ['voltage', 'current', 'imbalance', 'activation'],
              'voltage_weight': 1,
              'current_weight': 0.01,
              'imbalance_weight': 1e-5,
              'activation_weight': 1e-1,
              'forecast_horizon': 4,
              'flexibility': 0.1,
              'solar_scale': 0.8,
              'demand_scale': 10,
              'state_space': ['sun', 'demand', 'bus', 'imbalance'],
              'v_upper': 1.05,
              'v_lower': 0.95,
              'i_upper': 90,
              'demand_std': 0.03,
              'solar_std': 0.03,
              'total_imbalance': False,
              'reactive_power': True,
              'imbalance_change': False,
              'one_action':False}

    def set_parameters(self, new_parameters):
        """
        sets params for animals
        :param new_parameters: New parameter value
        :type new_parameters: dictionary
        """
        allowed_keys = ['episode_length', 'reward_terms', 'voltage_weight',
                        'current_weight', 'imbalance_weight',
                        'forecast_horizon', 'activation_weight', 'flexibility',
                        'state_space', 'solar_scale', 'demand_scale', 'v_lower',
                        'v_upper', 'i_upper', 'demand_std', 'solar_std',
                        'total_imbalance', 'reactive_power', 'imbalance_change',
                        'one_action']
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
        state_vars = ['state_space','total_imbalance','forecast_horizon']
        if any([var in new_parameters for var in state_vars]):
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                shape=self._get_obs().shape,
                                                dtype=np.float32)
        if 'one_action' in new_parameters:
            self.action_space = self.create_action_space()

        _ = self.reset(reset_time=False)

    def __init__(self, do_action=True, force_commitments=False, seed=None):
        self.np_random = None
        self._seed = self.seed(seed)
        self.do_action = do_action

        # power grid
        self.base_powergrid = cigre_network()
        pp.runpp(self.base_powergrid)
        self.powergrid = copy.deepcopy(self.base_powergrid)
        self.flexible_load_indices = np.arange(len(self.powergrid.load))
        self.last_action = np.zeros_like(self.flexible_load_indices)
        self.pq_ratio = self.calc_pq_ratio()

        # state variables, forecast + commitment
        self.solar_data = self.load_solar_data()
        self.demand_data = self.load_demand_data()
        # time attributes
        self._current_step = 0
        self._episode_start_hour = self.select_start_hour()
        self._episode_start_day = self.select_start_day()


        self.solar_forecasts = self.get_episode_solar_forecast()
        self.demand_forecasts = self.get_episode_demand_forecast()




        # self.set_demand_and_solar()
        self.force_commitments = force_commitments
        self._commitments = np.zeros(len(self.powergrid.load)) != 0
        self.resulting_demand = np.zeros(self.params['episode_length'])
        self._imbalance = self.empty_imbalance()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=self._get_obs().shape,
                                            dtype=np.float32)
        self.action_space = self.create_action_space()

        self.load_dict = self.get_load_dict()

    def get_load_dict(self):
        """
        :return: dictionary mapping columns name to index
        """
        return {col: index for (index, col) in enumerate(self.powergrid.load)}

    def create_action_space(self):
        if self.params['one_action']:
            action_space = spaces.Box(-1., 1., shape=(1,),
                       dtype=np.float32)
        else:
            action_space = spaces.Box(-1., 1., shape=self.last_action.shape,
                       dtype=np.float32)
        return action_space

    def calc_pq_ratio(self):
        """
        Power factor for loads are assumed constant.
        This method finds the PQ-ratio for all loads
        (same as for default cigre network)
        """
        net = create_cigre_network_mv(with_der="pv_wind")
        pq_ratio = net.load['q_mvar'] / net.load['p_mw']
        return pq_ratio

    def get_demand_forecast(self):
        """
        Finds the forecasted hourly demand for the next T hours
        :return: array with T demand for all buses
        """
        t = self._current_step
        horizon = self.params['forecast_horizon']
        return self.demand_forecasts[:, t:t + horizon]

    def get_solar_forecast(self):
        """
        Returns solar forecast for the next look_ahead hours.
        :return:
        """
        t = self._current_step
        horizon = self.params['forecast_horizon']
        return self.solar_forecasts[:, t:t + horizon]

    def get_scaled_solar_forecast(self):
        """
        scales each solar panel production with nominal values.
        :return: Sum of solar production in network (MW)
        """
        return self.solar_forecasts*self.powergrid.sgen['sn_mva'].sum()


    def get_scaled_demand_forecast(self):
        return self.demand_forecasts*self.powergrid.load['sn_mva'].sum()



    def select_start_hour(self):
        """
        Selects start hour for the episode
        :return:
        """
        return self.np_random.choice(24)

    def select_start_day(self):
        """
        Selects start day (date) for the data in the episode
        :return:
        """
        try:
            demand_data = self.demand_data
        except AttributeError:
            demand_data = self.load_demand_data()

        demand_days = (demand_data.index[-1] - demand_data.index[0])
        demand_days = demand_days.days
        episode_days = (self.params['episode_length'] // 24) + 1  # margin
        return self.np_random.choice(demand_days - episode_days)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, reset_time=True):
        self._current_step = 0

        self.powergrid = copy.deepcopy(self.base_powergrid)
        if reset_time:
            self._episode_start_hour = self.select_start_hour()
            self._episode_start_day = self.select_start_day()

        self.solar_forecasts = self.get_episode_solar_forecast()
        self.demand_forecasts = self.get_episode_demand_forecast()
        # self.set_demand_and_solar(reset_time=reset_time)
        self._imbalance = self.empty_imbalance()
        self.resulting_demand = np.zeros(self.params['episode_length'])

        return self._get_obs()

    def empty_imbalance(self):
        """
        Creates imbalance array, all 0.
        :return:
        """
        nr_loads = len(self.flexible_load_indices)
        episode_length = self.params['episode_length']
        return np.zeros((nr_loads, episode_length))

    def get_bus_state(self):
        """
        Return the voltage, active and reactive power at every bus
        :return:
        """

        return self.powergrid.res_bus.values.flatten()

    def get_episode_solar_forecast(self):
        """
        Method that returns the solar_forecast for the entire episode
        """
        episode_length = self.params['episode_length']
        horizon = self.params['forecast_horizon']
        start = self._episode_start_hour + self._episode_start_day * 24
        nr_hours = episode_length + horizon + 1  # margin of 1
        episode_solar_forecast = self.solar_data[start:start + nr_hours].values
        scaled_solar_forecast = episode_solar_forecast.ravel() * self.params['solar_scale']
        return scaled_solar_forecast.reshape(1, nr_hours)

    def get_episode_demand_forecast(self):
        """
        gets the forecasts for all loads in the episode
        :return:
        """
        episode_length = self.params['episode_length']
        horizon = self.params['forecast_horizon']
        demand_scale = self.params['demand_scale']

        start = self._episode_start_hour + self._episode_start_day * 24
        nr_hours = episode_length + horizon + 1  # margin of 1
        episode_demand_forecast = self.demand_data[start:start + nr_hours].values
        return (episode_demand_forecast.ravel() * demand_scale).reshape(1, nr_hours)

    def get_commitment_state(self):
        """
        Transforms _commitments array from booleans to 0,1
        :return:
        """
        commitments = np.zeros(self._commitments.shape)
        commitments[self._commitments] = 1
        return commitments

    def load_solar_data(self):
        solar_path = os.path.join(DATA_PATH, 'hourly_solar_data.csv')
        solar = pd.read_csv(solar_path)
        solar.index = pd.to_datetime(solar.iloc[:, 0])
        solar.index.name = 'time'
        solar = solar.iloc[:, [1]]

        return solar

    def load_demand_data(self):
        demand_path = os.path.join(DATA_PATH, 'hourly_demand_data.csv')
        demand = pd.read_csv(demand_path)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = 'time'
        demand = demand.iloc[:, [1]]

        return demand

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
            demand = demand_forecasts.ravel()
            state += list(demand)

        if 'sun' in self.params['state_space']:
            solar_forecasts = self.get_solar_forecast()
            state += list(solar_forecasts.ravel())

        if 'bus' in self.params['state_space']:
            bus_state = self.get_bus_state()
            state += list(bus_state)

        if self.force_commitments:
            commitment_state = self.get_commitment_state()
            state += list(commitment_state)

        if 'imbalance' in self.params['state_space']:
            net_load = self.powergrid.load['sn_mva'].sum()
            imbalance = self.calc_imbalance() / net_load
            if self.params['total_imbalance']:
                state += [imbalance.sum()]
            else:
                state += list(imbalance)

        return np.array(state)

    def calc_imbalance(self):
        """
        Calculates how much power the agent ows to the system, i.e the amount
        of extra energy the loads have received the last 24 hours.
        Reward function penalises a large balance.
        :return:

        """
        t = self._current_step
        if t > 24:
            modifications = self._imbalance[:, t - 24:t]

        elif t > 0:
            modifications = self._imbalance[:, :t]

        else:
            modifications = np.zeros((len(self.flexible_load_indices), 1))

        return modifications.sum(axis=1)

    def log_resulting_demand(self):
        """
        Logs the resulting demand in an episode after the agent has taken
        its actions

        """
        loads = self.powergrid.load['p_mw']
        self.resulting_demand[self._current_step] = loads.sum()

    def _take_action(self, action):
        """
        Takes the action vector, scales it and modifies the flexible loads
        :return:
        """
        nominal_load = self.powergrid.load['sn_mva']
        forecasted_load = nominal_load * self.get_demand_forecast()[:, 0]

        action *= self.params['flexibility'] * forecasted_load
        if self.force_commitments:
            action = self._check_commitment(action)

        self._imbalance[:, self._current_step] = action

        p_index = self.load_dict['p_mw']
        q_index = self.load_dict['q_mvar']

        self.powergrid.load.iloc[self.flexible_load_indices, p_index] += action
        if self.params['reactive_power']:
            self.powergrid.load.iloc[
                self.flexible_load_indices, q_index] += action * self.pq_ratio

        try:
            pp.runpp(self.powergrid)
            self.log_resulting_demand()
            return False

        except ppException:
            return True

    def set_demand_and_solar(self, reset_time=True):
        """
        Updates the demand and solar production according to the forecasts,
        with some noise.
        """
        net = self.powergrid
        solar_pu = copy.copy(self.get_solar_forecast()[:, 0])
        if reset_time:
            solar_pu += solar_pu * self.params['solar_std'] * self.np_random.randn()

        demand_pu = copy.copy(self.get_demand_forecast()[:, 0])
        if reset_time:
            demand_pu += demand_pu * self.params['demand_std'] * self.np_random.randn()

        net.sgen['p_mw'] = - solar_pu * net.sgen['sn_mva']
        net.load['p_mw'] = demand_pu * net.load['sn_mva']
        net.load['q_mvar'] = net.load['p_mw'] * self.pq_ratio

    def calc_reward(self, old_imbalance, action, include_loss=False):

        state_loss = 0
        if 'voltage' in self.params['reward_terms']:
            v = self.powergrid.res_bus['vm_pu']
            v_min = self.params['v_lower']
            v_max = self.params['v_upper']
            v_lower = sum(v_min - v[v < v_min]) * self.params['voltage_weight']
            v_over = sum(v[v > v_max] - v_max) * self.params['voltage_weight']
            state_loss += (v_lower + v_over)

        if 'current' in self.params['reward_terms']:
            i = self.powergrid.res_line['loading_percent']
            i_max = self.params['i_upper']
            i_over = sum(i[i > i_max] - i_max) * self.params['current_weight']
            state_loss += i_over

        if 'imbalance' in self.params['reward_terms']:
            balance = self.calc_imbalance()
            weight = self.params['imbalance_weight']
            if self.params['imbalance_change']:
                balance_change = np.abs(balance) - np.abs(old_imbalance)
                state_loss += balance_change.sum() * weight
            else:
                state_loss += np.abs(balance).sum() * weight

        if 'activation' in self.params['reward_terms']:
            action *= self.params['flexibility'] * self.powergrid.load['p_mw']
            act_loss = np.abs(action).sum() * self.params['activation_weight']
            state_loss += act_loss

        if include_loss:
            i_loss = sum(self.powergrid.res_line['pl_mw'])
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
        plt.plot(sol[0,:hours], axes=ax)
        plt.plot(load[0,:hours], axes=ax)
        plt.plot(resulting_demand[:hours], axes=ax)
        plt.legend(['solar', 'demand', 'modified'])
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
    location = 'C:\\Users\\vegar\\Dropbox\\Master\\thesis.git\\rl_power\\models\\'
    params_name = model_name +'_params.p'
    model = DDPG.load(location + model_name)
    env = ActiveEnv(seed=seed)
    with open(location + params_name,'rb') as f:
        params = pickle.load(f)

    env.set_parameters(params)
    model.set_env(env)
    return model, env




def calc_hour(start_hour, time_step):
    return (start_hour + time_step) % 24


def run_model():
    reward = 'voltage'
    period = 100000
    env = ActiveEnv()
    env.set_parameters({'reward_terms': [reward],
                        'demand_std': 0.03,
                        'solar_std': 0.03,
                        'reactive_power': True,
                        'state_space': ['sun', 'demand', 'imbalance']})
    rewards, t_steps, hues, hours = [], [], [], []
    obs = env.reset()

    env2 = copy.deepcopy(env)
    env2.do_action = False
    sol = env.solar_forecasts
    demand = env.demand_forecasts[0]

    show_sun, show_demand = True, True
    for t_step in range(1, period):

        action = np.ones(18)
        obs1, reward1, dones1, info1 = env.step(action)
        obs2, reward2, dones2, info2 = env2.step(action)

        current_step = env._current_step
        hour = calc_hour(env._episode_start_hour, current_step)

        if current_step == 198:
            print('noe')

        if current_step == 0:
            sol = env.solar_forecasts
            demand = env.demand_forecasts[0]

        rewards.append(reward1)
        hues.append('Agent')
        t_steps.append(t_step)
        hours.append(hour)

        rewards.append(reward2)
        hues.append('No agent')
        t_steps.append(t_step)
        hours.append(hour)

        if show_sun:
            rewards.append(sol[env._current_step - 1])
            hues.append('Sun')
            t_steps.append(t_step)
            hours.append(hour)

        if show_demand:
            rewards.append(demand[env._current_step - 1])
            hues.append('Demand')
            t_steps.append(t_step)
            hours.append(hour)


if __name__ == '__main__':

    env = ActiveEnv(force_commitments=False)
    env.set_parameters({'demand_std': 0})
    hours = 100
    for hour in range(hours):
        action = 1 * np.ones(len(env.powergrid.load))
        ob, reward, episode_over, info = env.step(action)

    total_demand = env.get_scaled_demand_forecast()[:hours].sum()
    total_modified_demand = env.resulting_demand[:hours].sum()
    assert np.abs(total_demand - total_modified_demand) < 10e-6

    print('haha')

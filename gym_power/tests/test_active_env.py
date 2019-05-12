# -*- coding: utf-8 -*-

"""

"""
import copy

import pytest
import numpy as np
from numpy.linalg import norm
from gym_power.envs.active_network_env import ActiveEnv

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'


ENV = ActiveEnv()


class TestForecasts:

    def test_initial_forecasts(self):
        """
        Checks that there are forecasts for every load, and that there always
        exist forecasts k hours in the future for all time steps.
        """
        episode_load_forecasts = ENV.get_episode_demand_forecast()
        assert len(episode_load_forecasts) == 1#len(ENV.powergrid.load)
        horizon = ENV.params['forecast_horizon']
        episode_length = ENV.params['episode_length']
        for load in episode_load_forecasts:
            assert load.shape[0] - episode_length >= horizon

    def test_daily_forecast(self):
        """
        Checks that daily forecast exists for all loads, and that there are 24
        hours for each load.
        :return:
        """
        daily_demand_forecast = ENV.get_demand_forecast()
        assert len(daily_demand_forecast) == 1#len(ENV.powergrid.load)
        horizon = ENV.params['forecast_horizon']

        for load in daily_demand_forecast:
            assert load.shape[0] == horizon

    def test_episode_solar_forecast(self):
        """Checks that end of the episode at least has a 24 hour forecast"""
        episode_solar_forecasts = ENV.get_episode_solar_forecast()
        horizon = ENV.params['forecast_horizon']
        episode_length = ENV.params['episode_length']

        assert len(episode_solar_forecasts) - episode_length> horizon

    def test_daily_solar_forecast(self):
        """
        checks that daily_solar_forecast is forecast_horizon hours
        """
        daily_solar_forecast = ENV.get_solar_forecast()
        horizon = ENV.params['forecast_horizon']

        assert len(daily_solar_forecast) == horizon


    def test_initial_state(self):
        """
        Checks that initial demand is equal to the forecast at the each bus
        (arbitrary definition)
        :return:
        """
        env = ActiveEnv()
        env.set_parameters({'demand_std':0})
        loads = env.powergrid.load['p_kw']
        demand_forecast = []
        for load in env.demand_forecasts:
            demand_forecast.append(load[0])
        demand_forecast = np.array(demand_forecast)
        demand_forecast *= env.powergrid.load['sn_kva']

        assert norm(demand_forecast - loads) < 10e-6

    def test_forecast_shift(self):
        """
        Checks that the forecast is shifted 1 timestep when step is called
        """
        env = ActiveEnv()
        start_solar = env.get_solar_forecast()
        start_demand = env.get_demand_forecast()
        action = env.action_space.sample()
        ob, reward, episode_over, info = env.step(action)
        assert norm(env.get_solar_forecast()[:-1] - start_solar[1:]) < 10e-5
        assert norm(env.get_demand_forecast()[0][:-1] - start_demand[0][1:]) < 10e-5

class TestState:

    def test_initial_loads(self):
        """
        Checks that set_demand_and_solar is called and updates the loads of
        the grid when instantiated. Also checks that demand is not altered
        if set_demand_and_solar is called when timestep is the same.
        """
        env = ActiveEnv()
        env.set_parameters({'demand_std':0})
        initial_loads = copy.copy(env.powergrid.load)
        env.set_demand_and_solar()
        loads = env.powergrid.load
        assert norm(initial_loads['p_kw'] - loads['p_kw']) < 10e-4
        assert norm(initial_loads['q_kvar'] - loads['q_kvar']) < 10e-4

    def test_error_term_solar(self):
        """
        Checks that the forecast values for solar irradiance deviates from
        the actual values
        """
        env = ActiveEnv()
        env.set_parameters({'solar_std': 0.1})
        while env.get_solar_forecast()[0] < 0.01: # to avoid night (no sun)
            action = env.action_space.sample()
            env.step(action)

        nominal_sun = env.powergrid.sgen['sn_kva']
        solar_forecast = nominal_sun*env.get_solar_forecast()[0]
        solar = -env.powergrid.sgen['p_kw']
        assert norm(solar_forecast- solar) > 0.1

    def test_error_term_demand(self):
        """
        Checks that the forecast values for solar irradiance deviates from
        the actual values
        """
        env = ActiveEnv()
        env.set_parameters({'demand_std': 0.1})
        nominal_load = env.powergrid.load['sn_kva']
        demand_forecast = nominal_load*env.get_demand_forecast()[0][0]
        demand = env.powergrid.load['p_kw']
        assert norm(demand - demand_forecast) > 0.1





class TestComitments:

    def test_action_override(self):
        """
        checks that actions of commited loads are overwritten and that commitments
        are updated correctly
        :return:
        """
        env = ActiveEnv(force_commitments=True)
        env._commitments[0] = True
        env.last_action[0] = 2
        action = np.ones_like(env.last_action)
        action[-1] = 0
        action = env._check_commitment(action)
        assert action[0] == -2
        assert all(env._commitments[1:-1])
        assert not any(env._commitments[[0,-1]])


class TestActions:

    def test_action(self):
        """
        Checks that action is taken and updates the network, but only if
        load is not commited
        :return:
        """
        flex = 0.1
        env = ActiveEnv(force_commitments=True)
        env.set_parameters({'flexibility':flex,
                            'demand_std':0})
        env.set_demand_and_solar()
        demand = copy.copy(env.powergrid.load['p_kw'].values)
        action1 = np.ones_like(env.last_action)
        action1 = env.action_space.sample()
        scaled_action1 = flex * action1 * env.powergrid.load['p_kw']

        env._take_action(action1)

        assert norm(env.powergrid.load['p_kw'].values - (
                    demand + scaled_action1)) < 10e-4
        action2 = env.action_space.sample()
        env._take_action(action2)

        # action2 should by modified to cancel effect of action1
        assert norm(env.last_action + scaled_action1) < 10e-4
        assert norm(env.powergrid.load['p_kw'].values - demand) < 10e-4

    def test_set_solar(self):
        """
        Checks that the the solar power production updates in every step,
        and follows the solar forecast
        :return:
        """
        env = ActiveEnv()
        env.set_parameters({'solar_std':0})
        solar_forecast = env.get_solar_forecast()
        for hour in range(4):
            action = env.action_space.sample()
            ob, reward, episode_over, info = env.step(action)
            load_pu = -env.powergrid.sgen['p_kw'] / env.powergrid.sgen['sn_kva']
            assert norm(load_pu - solar_forecast[hour]) < 10e-7


    def test_constant_consumption(self):
        """
        Checks that the total demand and total modified demand is the same.
        In other words, checks that consumption is shifted and not altered.
        This should hold when 'hours' is even
        """
        env = ActiveEnv(force_commitments=True)
        env.set_parameters({'demand_std':0})
        hours = 100
        for hour in range(hours):
            action = 1 * np.ones(len(env.powergrid.load))
            ob, reward, episode_over, info = env.step(action)

        total_demand = env.get_scaled_demand_forecast()[:hours].sum()
        total_modified_demand = env.resulting_demand[:hours].sum()
        assert np.abs(total_demand-total_modified_demand) < 10e-6
class TestReset:

    def test_reset_after_steps(self):
        """
        Checks that loads and solar production is reset.
        :return:
        """
        env = ActiveEnv()
        for hour in range(10):
            action = env.action_space.sample()
            ob, reward, episode_over, info = env.step(action)

        net = copy.copy(env.powergrid)
        env.reset()
        vars = ['p_kw', 'q_kvar']
        assert np.linalg.norm(
            net.load.loc[:, vars] - env.powergrid.load.loc[:, vars]) > 0.001
        assert np.linalg.norm(
            net.sgen.loc[:, vars] - env.powergrid.sgen.loc[:, vars]) > 0.001

    def test_reset_forecasts(self):
        """
        Checks that episode forecasts is reset
        :return:
        """
        env = ActiveEnv()
        start_env = copy.deepcopy(env)
        env.reset()

        assert norm(start_env.solar_forecasts - env.solar_forecasts) > 0.001
        assert norm(start_env.demand_forecasts[0][:24] - env.demand_forecasts[0][:24]) > 0.001


    def test_reset_episode_start_hour(self):
        """
        checks that _episode_start_hour is reset
        :return:
        """
        env = ActiveEnv()
        env._episode_start_hour = 100
        env.reset()
        assert env._episode_start_hour != 100

class TestSeeding:

    def test_forecast_seed(self):
        """
        Checks that same forecasts is used for same seed
        """
        env1 = ActiveEnv(seed=3)
        env2 = ActiveEnv(seed=3)
        env3 = ActiveEnv(seed=4)

        solar1 = env1.get_episode_solar_forecast()
        solar2 = env2.get_episode_solar_forecast()
        solar3 = env3.get_episode_solar_forecast()
        assert all(solar1 == solar2)
        assert any(solar1 != solar3)

        demand1 = env1.get_episode_demand_forecast()
        demand2 = env2.get_episode_demand_forecast()
        demand3 = env3.get_episode_demand_forecast()
        assert all(demand1[0] == demand2[0])
        assert any(demand1[0] != demand3[0])

    def test_set_params(self):
        """
        Checks that the forecasts are equal when set_parameters is used
        to change solar and demand scale
        """
        env1 = ActiveEnv(seed=3)
        solar_scale1 = env1.params['solar_scale']
        demand_scale1 = env1.params['demand_scale']


        env2 = ActiveEnv(seed=3)
        env2.set_parameters({'solar_scale':solar_scale1*2,
                             'demand_scale':demand_scale1*3})

        solar1, solar2 = env1.get_solar_forecast(), env2.get_solar_forecast()
        demand1, demand2 = env1.get_demand_forecast(), env2.get_demand_forecast()

        assert norm(solar1 * 2 - solar2) < 10e-7
        assert norm(demand1[0] * 3 - demand2[0]) < 10e-7





class TestParameters:

    def test_set_parameters(self):
        """
        Checks that observation space is updated if state_space parameters
        are updated
        """
        env = ActiveEnv()
        env.set_parameters({'state_space': ['sun', 'demand', 'imbalance','bus']})
        state0 = env.observation_space.shape[0]
        env.set_parameters({'state_space': ['sun', 'demand', 'imbalance']})
        state1 = env.observation_space.shape[0]
        assert state1 < state0


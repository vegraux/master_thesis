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


class TestForecasts():

    def test_initial_forecasts(self):
        """
        checks that there are forecasts for every load, and that there always
        exist forecasts 24 hours in the future for all time steps.
        """
        episode_load_forecasts = ENV.get_episode_demand_forecast()
        assert len(episode_load_forecasts) == len(ENV.powergrid.load)
        for load in episode_load_forecasts:
            assert load.shape[0] - ENV.episode_length > 24

    def test_daily_forecast(self):
        """
        Checks that daily forecast exists for all loads, and that there are 24
        hours for each load.
        :return:
        """
        daily_demand_forecast = ENV.get_demand_forecast()
        assert len(daily_demand_forecast) == len(ENV.powergrid.load)

        for load in daily_demand_forecast:
            assert load.shape[0] == 24

    def test_episode_solar_forecast(self):
        """Checks that end of the episode at least has a 24 hour forecast"""
        episode_solar_forecasts = ENV.get_episode_solar_forecast()
        assert len(episode_solar_forecasts) - ENV.episode_length > 24

    def test_daily_solar_forecast(self):
        """
        checks that daily_solar_forecast is 24 hours
        """
        daily_solar_forecast = ENV.get_solar_forecast()
        assert len(daily_solar_forecast) == 24


    def test_initial_state(self):
        """
        Checks that initial demand is equal to the forecast at the each bus
        (arbitrary definition)
        :return:
        """
        ENV._create_initial_state()
        loads = ENV.powergrid.load['p_kw']
        demand_forecast = []
        for load in ENV.demand_forcasts:
            demand_forecast.append(load[0])
        demand_forecast = np.array(demand_forecast)

        assert norm(demand_forecast - loads) < 10e-6


class TestState:

    def test_state_dimensions(self):
        """
        Checks that state dimensions are good. It will probably fail when I
        change state representation, so I guess it will be deleted at some point

        state = ENV._get_obs()
        state_size = len(state)
        bus_state_size = len(ENV.powergrid.bus) * 4
        demand_forecast = ENV.get_demand_forecast()
        demand_forecast_state_size = len(demand_forecast)*len(demand_forecast[0])
        solar_forecast_state_size =  len(ENV.get_solar_forecast())
        sum_size = bus_state_size + solar_forecast_state_size + demand_forecast_state_size
        assert state_size == sum_size

        """
        pass

class TestComitments:

    def test_action_override(self):
        """
        checks that actions of commited loads are overwritten and that commitments
        are updated correctly
        :return:
        """
        env = ActiveEnv()
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
        env = ActiveEnv()
        env.set_demand_and_solar()
        flex = 0.1
        demand = copy.copy(env.powergrid.load['p_kw'].values)
        action1 = np.ones_like(env.last_action)
        action1 = env.action_space.sample()
        scaled_action1 = flex * action1 * env.powergrid.load['p_kw']

        env._take_action(action1, flexibility=flex)

        assert norm(env.powergrid.load['p_kw'].values - (
                    demand + scaled_action1)) < 10e-4
        action2 = env.action_space.sample()
        env._take_action(action2)

        # action2 should by modified to cancel effect of action1
        assert norm(env.last_action + scaled_action1) < 10e-4
        assert norm(env.powergrid.load['p_kw'].values - demand) < 10e-4

    def test_set_solar(self):
        """
        Checks that the the solar power production is updates in every step,
        and follows the solar forecast
        :return:
        """
        env = ActiveEnv()
        solar_forecast = env.get_solar_forecast()
        for hour in range(10):
            action = env.action_space.sample()
            ob, reward, episode_over, info = env.step(action)
            load_pu = -env.powergrid.sgen['p_kw'] / env.powergrid.sgen['sn_kva']
            assert norm(load_pu - solar_forecast[hour]) < 10e-7




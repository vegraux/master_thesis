# -*- coding: utf-8 -*-

"""

"""
import pytest
import numpy as np
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
        episode_load_forecasts = ENV.get_episode_load_forecast()
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

        assert np.linalg.norm(demand_forecast - loads) < 10e-6


class TestState:

    def test_state_dimensions(self):
        """
        Checks that state dimensions are good. It will probably fail when I
        change state representation, so I guess it will be deleted at some point
        """
        state = ENV._get_obs()
        state_size = len(state)
        bus_state_size = len(ENV.powergrid.bus) * 4
        demand_forecast = ENV.get_demand_forecast()
        demand_forecast_state_size = len(demand_forecast)*len(demand_forecast[0])
        solar_forecast_state_size =  len(ENV.get_solar_forecast())
        sum_size = bus_state_size + solar_forecast_state_size + demand_forecast_state_size
        assert state_size == sum_size


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
        initial_forecasts = ENV.get_initial_load_forecast()
        assert len(initial_forecasts) == len(ENV.powergrid.load)
        for load in initial_forecasts:
            assert load.shape[0] - ENV.episode_length > 24

    def test_daily_forecast(self):
        """
        cheks that daily forecast exists for all loads, and that there are 24
        hours for each load.
        :return:
        """
        daily_forecast = ENV.get_forecast()
        assert len(daily_forecast) == len(ENV.powergrid.load)

        for load in daily_forecast:
            assert load.shape[0] == 24


    def test_initial_state(self):
        ENV._create_initial_state()
        loads = ENV.powergrid.load['p_kw']
        forecasted_loads = []
        for load in ENV.load_forcasts:
            forecasted_loads.append(load[0])
        forecasted_loads = np.array(forecasted_loads)

        assert np.linalg.norm(forecasted_loads - loads) < 0.0001



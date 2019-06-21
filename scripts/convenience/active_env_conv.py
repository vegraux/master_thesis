# -*- coding: utf-8 -*-

"""

"""
from active_env.envs.active_network_env import ActiveEnv
import matplotlib.pyplot as plt
__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'


def plot_forecasts():
    """
    Plots solar and load forecast.
    :return:
    """
    hours = 100
    env = ActiveEnv()
    load = env.get_episode_demand_forecast()[0]
    sol = env.get_episode_solar_forecast()
    fig, ax = plt.subplots()

    plt.plot(2000*sol[:hours],axes=ax)
    plt.plot(load[:hours],axes=ax)
    plt.show()

if __name__ == '__main__':
    plot_forecasts()
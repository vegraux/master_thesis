# -*- coding: utf-8 -*-

"""

"""
from gym_power.envs.active_network_env import ActiveEnv
import matplotlib.pyplot as plt
__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'


def plot_forecasts():
    """
    Plots solar and load forecast.
    :return:
    """
    env = ActiveEnv()
    load = env.get_episode_load_forecast()[0]
    sol = env.get_episode_solar_forecast()
    fig, ax = plt.subplots()

    plt.plot(sol[:100],axes=ax)
    plt.plot(5*load[:100],axes=ax)
    plt.show()

if __name__ == '__main__':
    plot_forecasts()
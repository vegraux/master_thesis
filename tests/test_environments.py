# -*- coding: utf-8 -*-
import pytest
import pandas as pd
import pandapower as pp
from power_env.envs.twobus_env import PowerEnvOld
"""

"""

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'


class TestPowerEnv:
    """
    :type env: PowerEnvOld
    """

    def test_calc_reward(self):
        env = PowerEnvOld()
        env.reset()
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        env.close()

        assert env.observation_space.contains(obs)
        assert reward < 0


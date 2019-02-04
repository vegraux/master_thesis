# -*- coding: utf-8 -*-

"""

"""
from gym.envs.registration import register

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'


register(
    id='foo-v0',
    entry_point='gym_power.envs:PowerEnv',
)

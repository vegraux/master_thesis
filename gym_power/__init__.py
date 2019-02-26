# -*- coding: utf-8 -*-

"""

"""
from gym.envs.registration import register

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'


register(
    id='powerenv-v0',
    entry_point='gym_power.envs:PowerEnv',
)
register(
    id='powerenv_sparse-v0',
    entry_point='gym_power.envs:PowerEnvSparse',
)
register(
    id='powerenv_goal-v0',
    entry_point='gym_power.envs:PowerGoalEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20}

)

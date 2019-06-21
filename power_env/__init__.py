# -*- coding: utf-8 -*-

"""

"""
from gym.envs.registration import register

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'


register(
    id='active_env-v0',
    entry_point='power_env.envs:ActiveEnv'
)

register(
    id='twobus_env-v0',
    entry_point='power_env.envs:TwoBusEnv',
)
register(
    id='twobus_sparse_env-v0',
    entry_point='power_env.envs:TwoBusSparseEnv',
)
register(
    id='twobus_goal_env-v0',
    entry_point='power_env.envs:TwoBusGoalEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20}

)

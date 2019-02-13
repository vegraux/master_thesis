# -*- coding: utf-8 -*-

"""

"""

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'

from gym_power.envs import PowerEnv
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines import DDPG
import matplotlib.pyplot as plt
import pandas as pd


powerenv = PowerEnv()
powerenv = DummyVecEnv([lambda: powerenv])



powermodel = DDPG(MlpPolicy, powerenv, verbose=1,
                 tensorboard_log='C:\\Users\\vegar\\Dropbox\\Master\\logs',
                  memory_limit=int(1e5))

t_steps = 3000
powermodel.learn(t_steps)

load = [k[-2] for k in powermodel.memory.observations0.data[:t_steps]]
actions = pd.DataFrame(data=powermodel.memory.actions.data[:t_steps])


# -*- coding: utf-8 -*-

"""

"""

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'

from gym_power.envs import PowerEnv, PowerEnvSparse
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.ddpg.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf


#powerenv = PowerEnv()
powerenv = PowerEnvSparse()

powerenv = DummyVecEnv([lambda: powerenv])

action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0.]), sigma=np.array([0.05]))

param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.05)

t_steps = 3000
powermodel = DDPG(MlpPolicy, powerenv,
                  verbose=2,
                  #action_noise=action_noise,
                  param_noise=param_noise,
                  tensorboard_log='C:\\Users\\vegar\\Dropbox\\Master\\logs',
                  memory_limit=int(t_steps),
                  nb_train_steps=50,
                  nb_rollout_steps=100,
                  normalize_observations=True)

powermodel.learn(t_steps)

df = pd.DataFrame(powermodel.memory.observations0.data[:t_steps,[-3]],columns=['load'])
df['rewards'] = powermodel.memory.rewards.data[:t_steps]
df['actions'] = powermodel.memory.actions.data[:t_steps]*powermodel.action_space.high
df.plot()
plt.show()
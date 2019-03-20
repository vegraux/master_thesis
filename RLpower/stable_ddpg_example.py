# -*- coding: utf-8 -*-

"""

"""

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'

from gym_power.envs import PowerEnv, PowerEnvSparse
from gym_power.envs.power_env import PowerEnvOld
from gym_power.envs.active_network_env import ActiveEnv
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf


#powerenv = PowerEnvSparse()
#powerenv = PowerEnv()
powerenv = ActiveEnv()
#powerenv = PowerEnvOld()
powerenv = DummyVecEnv([lambda: powerenv])
powerenv = VecNormalize(powerenv, norm_reward=False)

#action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0]), sigma=np.array([0.3]))
action_mean = np.zeros(powerenv.action_space.shape)
action_sigma = 0.3 * np.ones(powerenv.action_space.shape)
action_noise = OrnsteinUhlenbeckActionNoise(mean=action_mean, sigma=action_sigma)

param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.01)

t_steps = 100000
powermodel = DDPG(LnMlpPolicy, powerenv,
                  verbose=2,
                  action_noise=action_noise,
                  #param_noise=param_noise,
                  tensorboard_log='C:\\Users\\vegar\\Dropbox\\Master\\logs',
                  memory_limit=int(t_steps),
                  nb_train_steps=50,
                  nb_rollout_steps=100,
                  critic_lr=0.00001, #default: 0.001
                  actor_lr=0.000001, #default: 0.0001
                  normalize_observations=True)
powermodel.learn(t_steps)


my_env = powermodel.env.venv.venv.envs[0]

df = pd.DataFrame(powermodel.memory.unnormalized_obs.data[:t_steps,[-2]],columns=['load'])
df['rewards'] = powermodel.memory.rewards.data[:t_steps]
df['actions'] = powermodel.memory.actions.data[:t_steps]*my_env.max_power
df.plot()
plt.show()

df2 = pd.DataFrame()

obs_det = []
rewards_det = []
action_det = []

obs = powerenv.reset()
for i in range(200):
    action,_ = powermodel.predict(obs)
    action_scaled = 0.5*(action+1)*powerenv.envs[0].max_power
    obs, rewards, dones, info = powerenv.step(action_scaled)
    obs_det.append(obs[0][-2])
    rewards_det.append(rewards[0])
    action_det.append(action_scaled[0][0])

df2['rewards'] = rewards_det
df2['load'] = obs_det
df2['actions'] = action_det
df2.plot()
plt.show()
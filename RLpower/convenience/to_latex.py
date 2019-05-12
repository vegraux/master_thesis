# -*- coding: utf-8 -*-

"""
Training the reinforcement agent using stable baselines
"""
__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'

from gym_power.envs.active_network_env import ActiveEnv
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec
import numpy as np

powerenv = ActiveEnv()
powerenv.set_parameters({'state_space': ['sun','demand','imbalance'],
                         'reward_terms':['voltage','current','imbalance']})

powerenv = DummyVecEnv([lambda: powerenv])
action_mean = np.zeros(powerenv.action_space.shape)
action_sigma = 0.3 * np.ones(powerenv.action_space.shape)
action_noise = OrnsteinUhlenbeckActionNoise(mean=action_mean,
                                            sigma=action_sigma)

param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.2,
                                     desired_action_stddev=0.01)

t_steps = 800000
logdir = 'C:\\Users\\vegar\\Dropbox\\Master\\logs'
powermodel = DDPG(LnMlpPolicy, powerenv,
                  verbose=2,
                  action_noise=action_noise,
                  gamma=0.99,
                  #param_noise=param_noise,
                  tensorboard_log=logdir,
                  memory_limit=int(800000),
                  nb_train_steps=50,
                  nb_rollout_steps=100,
                  critic_lr=0.001,
                  actor_lr=0.0001,
                  normalize_observations=False)
powermodel.learn(t_steps)




# -*- coding: utf-8 -*-

"""

"""

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'

import gym
import numpy as np
import sys
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import ACKTR

from stable_baselines import PPO2


import imageio
import numpy as np

from stable_baselines.common.policies import MlpPolicy, ActorCriticPolicy
from stable_baselines import A2C
from stable_baselines.her import HER
from stable_baselines.common.vec_env import DummyVecEnv

from gym_power.envs import PowerEnv

#MountainCarContinuous-v0

powerenv = PowerEnv()
powerenv = DummyVecEnv([lambda: powerenv])

model = A2C(MlpPolicy, "CartPole-v1",verbose=1)
model1 = HER(ActorCriticPolicy,'MountainCarContinuous-v0', verbose=1)
powermodel = A2C(MlpPolicy, powerenv, verbose=1,
                 tensorboard_log='C:\\Users\\vegar\\Dropbox\\Master\\logs')

model.learn(50000)

#model = HER(ActorCriticPolicy,'MountainCarContinuous-v0', verbose=1)
#model.learn(30000)

images = []
obs = model.env.reset()
img = model.env.render(mode='rgb_array')
for i in range(350):
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, _ ,_ = model.env.step(action)
    img = model.env.render(mode='rgb_array')

imageio.mimsave('cartpole_a2c.gif', [np.array(img[0]) for i, img in enumerate(images) if i%2 == 0], fps=29)
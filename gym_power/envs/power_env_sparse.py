# -*- coding: utf-8 -*-

"""

"""
from gym_power.envs.power_env import PowerEnv
from gym import error, spaces, utils
import numpy as np



__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'


class PowerEnvSparse(PowerEnv):
    """
    sparse rewards, with desired state in observation
    """
    def __init__(self):

        super(PowerEnvSparse, self).__init__()
        self.observation_size = 4 * len(
            self.powergrid.bus) + 1

        high = np.array([1000000 for _ in range(self.observation_size)])

        self.observation_space = spaces.Box(low=-high, high=high,
                                            dtype=np.float32)

    def calc_reward(self):
        flows = self.powergrid.res_bus
        reward = np.abs(flows.iloc[1, 2] - self.target_load) < 100  # + \
        return reward

    def step(self, action):
        episode_over = self._take_action(action)
        if not episode_over:
            reward = self._get_reward()
            ob = self._get_obs()

        else:
            reward = 0
            #ob = self.reset()
            ob = None
        return ob, reward, episode_over, {}

    def _get_obs(self):
        obs = list(self.powergrid.res_bus.values.flatten())
        obs.append(self.target_load)
        return np.array(obs)


if __name__ == '__main__':
    env = PowerEnvSparse()
    print('wait')
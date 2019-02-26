# -*- coding: utf-8 -*-

"""

"""

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'
import sys
import baselines.run as run

import gym_power

MODEL_PATH = '\\PycharmProjects\\highway-env\\scripts\\models\\latest'

DEFAULT_ARGUMENTS = [
    "--env=powerenv_goal-v0",
    "--alg=her",
    "--num_timesteps=1e4",
    "--network=default",
    "--num_env=0",
    "--save_video_interval=0",
    "--play"
]

if __name__ == "__main__":
    args = sys.argv
    if len(args) <= 1:
        args = DEFAULT_ARGUMENTS
    run.main(args)

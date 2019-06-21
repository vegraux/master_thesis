# -*- coding: utf-8 -*-

"""

"""

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'
import sys
import baselines.run as run

import power_env

MODEL_PATH = '\\PycharmProjects\\highway-powergrid\\scripts\\models\\latest'

DEFAULT_ARGUMENTS = [
    "--powergrid=powerenv_goal-v0",
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

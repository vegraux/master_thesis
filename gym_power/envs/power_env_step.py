# -*- coding: utf-8 -*-

"""

"""
from gym_power.envs.power_env import PowerEnvOld
from pandapower import ppException
import pandas as pd

__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'


class PowerEnvStep(PowerEnvOld):
    """
    The action taken is changes from the current situation, i.e the action
    by the agent is added to the current power production. The production is
    not abosolute, but relative.
    """
    def __init__(self):
        super(PowerEnvStep, self).__init__()

    def _take_action(self, action):
        """ Converts the action space into an pandapowe action. """
        at = pd.DataFrame(data=action, columns=['p_kw'])
        self.powergrid.gen[at.columns] += at
        try:
            pp.runpp(self.powergrid)
            return False

        except ppException:
            return True





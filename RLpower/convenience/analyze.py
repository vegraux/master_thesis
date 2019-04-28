# -*- coding: utf-8 -*-

"""

"""
from __future__ import print_function
import pandapower as pp
import os
import dotenv
dotenv.load_dotenv()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import stable_baselines
import pickle
from stable_baselines import DDPG
import sys
import copy
import dotenv
from  gym_power.envs.active_network_env import ActiveEnv
import seaborn as sns
__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'

MODEL_PATH = os.getenv('MODEL_PATH')

def find_load_names(sol_bus):
    nr_sol = 1
    nr_else = 1
    load_names = []
    for k in range(len(sol_bus)):
        if sol_bus[k]:
            load_names.append('sun {}'.format(nr_sol))
            nr_sol += 1
        else:
            load_names.append('load {}'.format(nr_else))
            nr_else += 1
    return load_names


def simulate_day2(env, model, show_imbalance=False, show_solar=True,
                  show_action=True,
                  show_demand=False, period=25):
    net = env.powergrid
    sol_bus = net.load['bus'].isin(net.sgen['bus'])
    actions = []
    t_steps = []
    flex_loads = []
    sols = []
    obs = env.reset()
    sol = env.solar_forecasts
    demand = env.demand_forecasts[0]
    names = find_load_names(sol_bus)
    hues = []
    for t_step in range(1, period):

        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        if show_action:
            actions += list(action)
            hues += ['action' for _ in range(len(action))]
            t_steps += list(t_step * np.ones_like(action))
            flex_loads += names

        if show_solar:
            actions += list(sol[t_step - 1] * np.ones_like(action))
            hues += ['sun' for _ in range(len(action))]
            t_steps += list(t_step * np.ones_like(action))
            flex_loads += names
        if show_imbalance:
            try:
                imbalance = env.calc_balance() / 30000
            except AttributeError:
                imbalance = env.calc_imbalance() / 30000
            actions += list(imbalance * np.ones_like(action))
            hues += ['imbalance' for _ in range(len(action))]
            t_steps += list(t_step * np.ones_like(action))
            flex_loads += names

        if show_demand:
            actions += list(demand[t_step - 1] * np.ones_like(action))
            hues += ['demand' for _ in range(len(action))]
            t_steps += list(t_step * np.ones_like(action))
            flex_loads += names

    df = pd.DataFrame()
    df['actions'] = actions
    df['steps'] = t_steps
    df['load'] = flex_loads
    df[''] = hues
    return df


def data_from_subplot(ax, imbalance=False):
    actions = ax.lines[3].get_ydata()
    sun = ax.lines[4].get_ydata()
    balance = ax.lines[5].get_ydata()
    oneplot = pd.DataFrame()
    if imbalance:
        oneplot['Energy imbalance'] = balance
    oneplot['Solar irradiance'] = sun
    oneplot['Action'] = actions
    return oneplot


def load_env(model_name='flexible_load_first',seed=9):
#flexible_load_first, overnight, larger_margin_cost, discount_06, flex50
    model_path = os.path.join(MODEL_PATH,model_name)
    params_name = model_name +'_params.p'
    param_path = os.path.join(MODEL_PATH,params_name)
    model = DDPG.load(model_path)
    env = ActiveEnv(seed=seed)
    with open(param_path,'rb') as f:
        params = pickle.load(f)

    env.set_parameters(params)
    model.set_env(env)
    return model, env


def resulting_voltage():
    period = 199
    model, env = load_env(seed=9)
    net = env.powergrid
    actions = []
    t_steps = []
    flex_loads = []
    obs = env.reset()
    sol = env.solar_forecasts
    hues = []
    env.set_parameters({'flexibility': 0.25,
                        'solar_scale': 1.2})
    env2 = copy.deepcopy(env)
    env2.do_action = False

    for t_step in range(1, period):
        # action,_ = model.predict(obs)
        action2 = -np.ones(18)
        obs1, rewards1, dones1, info1 = env.step(action2)

        obs2, rewards2, dones2, info2 = env2.step(action2)


        v_agent = env.powergrid.res_bus.vm_pu
        actions += list(v_agent)
        hues += ['-25% demand' for _ in range(len(v_agent))]
        t_steps += list(t_step * np.ones_like(v_agent))
        flex_loads += list(range(len(v_agent)))

        v_no_action = env2.powergrid.res_bus.vm_pu
        actions += list(v_no_action)
        hues += ['No agent' for _ in range(len(v_no_action))]
        t_steps += list(t_step * np.ones_like(v_no_action))
        flex_loads += list(range(len(v_agent)))

    df = pd.DataFrame()
    df['Voltage [pu]'] = actions
    df['Hours'] = t_steps
    df['Load'] = flex_loads
    df[''] = hues
    return df


def change_hours(x,start=155):
    return x-start
def change_legend(x):
    if x == 'No agent':
        return 'No action'
    else:
        return x


def resulting_current():
    period = 199
    model, env = load_env(seed=8)
    net = env.powergrid
    actions = []
    t_steps = []
    flex_loads = []
    obs = env.reset()
    hues = []
    env.set_parameters({'solar_scale': 1.2,
                        'flexibility': 0.25})
    env2 = copy.deepcopy(env)
    env2.do_action = False

    for t_step in range(1, period):
        action, _ = model.predict(obs)
        action2 = np.ones(18)
        action2[[0, 10]] = -1

        obs1, rewards1, dones1, info1 = env.step(action2)
        obs2, rewards2, dones2, info2 = env2.step(action2)

        i_agent = env.powergrid.res_line['loading_percent']
        actions += list(i_agent)
        hues += ['+ 25 % demand' for _ in range(len(i_agent))]
        t_steps += list(t_step * np.ones_like(i_agent))
        flex_loads += list(range(len(i_agent)))

        i_no_agent = env2.powergrid.res_line['loading_percent']
        actions += list(i_no_agent)
        hues += ['No actions' for _ in range(len(i_no_agent))]
        t_steps += list(t_step * np.ones_like(i_no_agent))
        flex_loads += list(range(len(i_no_agent)))

    df = pd.DataFrame()
    df['Line capacity [%]'] = actions
    df['Hours'] = t_steps
    df['Line'] = flex_loads
    df[''] = hues
    return df

def plot_resulting_current(df):
    sns.set(style="ticks")
    grid = sns.FacetGrid(df, col="Line", hue="",
                         col_wrap=6, height=1.5)

    grid.map(plt.axhline, y=1, ls=":", c=".5")

    grid.map(plt.plot, "Hours", "Line capacity [%]")
    grid.add_legend()

def plot_resulting_voltage(df):
    sns.set(style="ticks")
    grid = sns.FacetGrid(df, col="Load", hue="",
                         col_wrap=6, height=1.5)

    grid.map(plt.axhline, y=1, ls=":", c=".5")

    grid.map(plt.plot, "Hours", "Voltage [pu]")
    grid.add_legend()


def reward_agent_vs_no_action(model_name='flexible_load_first'
                              ,period=600, show_sun=False, show_demand=False,
                              reward_terms =['voltage','current'],
                              seed=5):
    model, env = load_env(model_name= model_name, seed=seed)  # seed 5: heavy sun
    env.set_parameters({'reward_terms': reward_terms})
    rewards = []
    t_steps = []
    obs = env.reset()
    hues = []
    env2 = copy.deepcopy(env)
    env2.do_action = False
    sol = env.solar_forecasts
    demand = env.demand_forecasts[0]

    for t_step in range(1, period):

        action, _ = model.predict(obs)
        obs1, reward1, dones1, info1 = env.step(action)
        obs2, reward2, dones2, info2 = env2.step(action)
        if env._current_step == 0:
            sol = env.solar_forecasts
            demand = env.demand_forecasts[0]
            env2.demand_forecasts = env.demand_forecasts

        rewards.append(reward1)
        hues.append('Agent')
        t_steps.append(t_step)

        rewards.append(reward2)
        hues.append('No agent')
        t_steps.append(t_step)

        if show_sun:
            rewards.append(sol[env._current_step - 1])
            hues.append('Sun')
            t_steps.append(t_step)

        if show_demand:
            rewards.append(demand[env._current_step - 1])
            hues.append('Demand')
            t_steps.append(t_step)

    df = pd.DataFrame()
    df['Reward'] = rewards
    df['Hours'] = t_steps
    df[''] = hues
    return df

def update_param_dict(for_reals=False):
    if for_reals:
        env = ActiveEnv()
        model_dir = 'C:\\Users\\vegar\Dropbox\Master\\thesis.git\RLpower\models'
        for model in os.listdir(model_dir):
            if 'params' in model:
                with open(os.path.join(model_dir,model),'rb') as f:
                    olds_params = pickle.load(f)
                    missing_params = [p for p in env.params if p not in olds_params]
                    params_values = {'reactive_power':False,
                                     'solar_std': 0,
                                     'total_imbalance':True,
                                     'demand_std': 0}
                    for param in missing_params:
                        olds_params[param] = params_values[param]

                assert all([p in env.params for p in olds_params])
                with open(os.path.join(model_dir,model),'wb') as f:
                    pickle.dump(olds_params,f)


def agent_rewards(period=20000, model_name='flexible_load_first'):
    """
    Tests the agent in terms of current and voltage reward
    """
    model, env_voltage = load_env(model_name,
                          seed=9)  # seed 5: heavy sun, 9: weak sun

    rewards, t_steps, hues, hours = [], [], [], []
    obs = env_voltage.reset()
    env_voltage.set_parameters({'reward_terms': ['voltage']})
    env_current = copy.deepcopy(env_voltage)
    env_current.set_parameters({'reward_terms': ['current']})
    env_no_agent = copy.deepcopy(env_voltage)
    env_no_agent.do_action = False

    sol = env_voltage.solar_forecasts
    demand = env_voltage.demand_forecasts[0]

    show_sun, show_demand = True, True
    for t_step in range(1, period):
        action, _ = model.predict(obs)
        _, voltage_reward, _, _ = env_voltage.step(action)
        _, current_reward, _, _= env_current.step(action)
        _, no_agent_reward, _, _ = env_no_agent.step(action)

        if env_voltage._current_step == 0:
            sol = env_voltage.solar_forecasts
            demand = env_voltage.demand_forecasts[0]

        current_step = env_voltage._current_step
        hour = calc_hour(env_voltage._episode_start_hour,current_step)

        rewards.append(voltage_reward)
        hues.append('Voltage')
        t_steps.append(t_step)
        hours.append(hour)

        rewards.append(current_reward)
        hues.append('Current')
        t_steps.append(t_step)

        rewards.append(no_agent_reward)
        hues.append('No agent')
        t_steps.append(t_step)

        if show_sun:
            rewards.append(sol[current_step - 1])
            hues.append('Sun')
            t_steps.append(t_step)

        if show_demand:
            rewards.append(demand[current_step - 1])
            hues.append('Demand')
            t_steps.append(t_step)

    df = pd.DataFrame()
    df['Reward'] = rewards
    df['Hours'] = t_steps
    df[''] = hues
    return df

def calc_hour(start_hour,time_step):
    return (start_hour + time_step) % 24

def append_rewards(rewards, reward,hues,hue,t_steps,t_step,hours,hour):
    rewards.append(reward)
    hues.append(hue)
    t_steps.append(t_step)
    hours.append(hour)
    return rewards, hues, t_steps, hours


def simulate_day3(env, model, show_imbalance=False, show_solar=True,
                  show_action=True,
                  show_demand=False, period=25):
    """
    simulate grid and save info about action of each bus, hour of day etc.
    :param env:
    :param model:
    :param show_imbalance:
    :param show_solar:
    :param show_action:
    :param show_demand:
    :param period:
    :return:
    """
    net = env.powergrid
    actions, t_steps, flex_loads, sols, hours = [], [], [], [], []

    obs = env.reset()
    sol = env.solar_forecasts
    demand = env.demand_forecasts[0]
    hues = []
    for t_step in range(1, period):

        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        if show_action:
            actions += list(action)
            hues += ['action' for _ in range(len(action))]
            t_steps += list(t_step * np.ones_like(action))
            flex_loads += list(net.load.index)

        if show_solar:
            actions += list(sol[t_step - 1] * np.ones_like(action))
            hues += ['sun' for _ in range(len(action))]
            t_steps += list(t_step * np.ones_like(action))
            flex_loads += list(net.load.index)
        if show_imbalance:
            try:
                imbalance = env.calc_balance() / 30000
            except AttributeError:
                imbalance = env.calc_imbalance() / 30000
            actions += list(imbalance * np.ones_like(action))
            hues += ['imbalance' for _ in range(len(action))]
            t_steps += list(t_step * np.ones_like(action))
            flex_loads += list(net.load.index)

        if show_demand:
            actions += list(demand[t_step - 1] * np.ones_like(action))
            hues += ['demand' for _ in range(len(action))]
            t_steps += list(t_step * np.ones_like(action))
            flex_loads += list(net.load.index)

        hour = calc_hour(env._episode_start_hour, env._current_step)
        hours += [hour for _ in range(len(action)*3)]

    df = pd.DataFrame()
    df['actions'] = actions
    df['steps'] = t_steps
    df['load'] = flex_loads
    df['hour'] = hours
    df[''] = hues
    return df





if __name__ == '__main__':
    model, env = load_env()
    df = simulate_day3(env, model, show_demand=True, period=199)
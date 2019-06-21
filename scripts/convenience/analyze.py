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
import pandapower.networks as pn
from stable_baselines import DDPG, PPO1
import sys
import copy
import dotenv
from  active_env.envs.active_network_env import ActiveEnv
import seaborn as sns
__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'

MODEL_PATH = os.getenv('MODEL_PATH')
DATA_PATH = os.getenv('DATA_PATH')

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

def save_model(model,env,model_name):
    """
    Saves RL model, and parameters for environment
    :param model: Stable baselines model
    :param env: gym environment
    :param model_name: name of the model
    """
    path = 'models/' + model_name + '.pkl'
    i = 2
    while os.path.isfile(path):
        model_name += '_' + str(i)
        i += 1
        path = 'models/' + model_name + '.pkl'
    model.save('models/' + model_name)
    with open('models/' + model_name + '_params.p', 'wb') as f:
        pickle.dump(env.envs[0].params, f)

def load_env(model_name='flexible_load_first',seed=9):
#flexible_load_first, overnight, larger_margin_cost, discount_06, flex50
    model_path = os.path.join(MODEL_PATH,model_name)
    params_name = model_name +'_params.p'
    param_path = os.path.join(MODEL_PATH,params_name)
    try:
        model = DDPG.load(model_path)
    except:
        model = PPO1.load(model_path)
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


def peak_network(peak='demand', scale_nominal_value=40):
    """
    Creates 15 bus cigre network with 8 solar units and 1 wind park.
    The network is scaled so that there is a lot of solar power production
    :param with_der:
    :return:
    """

    net = pn.create_cigre_network_mv(with_der="pv_wind")
    pq_ratio = net.load['q_mvar'] / net.load['p_mw']

    net.sgen['sn_mva'] *= scale_nominal_value
    net.sgen.loc[
        8, 'sn_mva'] /= scale_nominal_value  # undo scaling for wind park
    if peak == 'solar':
        net.sgen['p_mw'] = net.sgen[
                               'sn_mva'].values * 0.794747  # max solar at 12 am
        net.load['p_mw'] = net.load['sn_mva'] * 0.444  # mean demand at 12 am
        net.load['q_mvar'] = net.load['p_mw'] * pq_ratio
    elif peak == 'demand':
        net.sgen['p_mw'] = net.sgen['sn_mva'] * 0.039228  # mean solar at 7 pm
        net.load['p_mw'] = net.load['sn_mva'] * 0.926647  # max demand at 7 pm
        net.load['q_mvar'] = net.load['p_mw'] * pq_ratio
    pp.runpp(net)
    return net

def current_effect(net=None, buses=[1,12], scale_factor=2):
    """
    Effect of scaling consumption at buses
    """
    if net is None:
        net = pn.create_cigre_network_mv(with_der="pv_wind")
        net.load[['p_mw','q_mvar']] = net.load[['p_mw','q_mvar']]*0.748772 #mean demand in hour 8
        pp.runpp(net)
    loading1 = net.res_line['loading_percent']
    idx = net.load['bus'].isin(buses)
    net.load.loc[idx,['p_mw','q_mvar']] *= scale_factor
    pp.runpp(net)
    loading2 = net.res_line['loading_percent']
    df = pd.DataFrame(data={'A':loading1,'B':loading2})

    return df, net

def voltage_effect(net=None, buses=[1,12], scale_factor=2):
    if net is None:
        net = pn.create_cigre_network_mv(with_der="pv_wind")
        net.load[['p_mw','q_mvar']] = net.load[['p_mw','q_mvar']]*0.748772 #mean demand at 20 pm
        pp.runpp(net)
    v1 = net.res_bus['vm_pu']
    idx = net.load['bus'].isin(buses)
    net.load.loc[idx,['p_mw','q_mvar']] *= scale_factor
    pp.runpp(net)
    v2 = net.res_bus['vm_pu']
    df = pd.DataFrame(data={'A':v1,'B':v2})
    return df, net


def decisive_bus(buses, base_net=None, kind='voltage', scale_factor=2):
    buses = range(1, 15)
    mean_diff = []
    if base_net is None:
        base_net = pn.create_cigre_network_mv(with_der="pv_wind")
        pp.runpp(base_net)

    for bus in buses:
        net = copy.deepcopy(base_net)
        if kind == 'voltage':
            df, net = voltage_effect(net=net, buses=[bus],
                                     scale_factor=scale_factor)
        elif kind == 'current':
            df, net = current_effect(net=net, buses=[bus],
                                     scale_factor=scale_factor)
        diff = df['A'] - df['B']
        mean_diff.append(diff.sum())
    diffs = pd.DataFrame(data=mean_diff, index=buses, columns=[kind])
    flex = '{:d} %'.format(int((scale_factor - 1) * 100))
    diffs['Flexibility'] = flex
    return diffs


def calc_impacts(kind='voltage', peak='demand', flex=0.2,
                 max_flex=False):
    """
    Calc voltage/current impact in either peak demand or peak solar for all buses
    """
    base_net = peak_network(peak)
    values = None
    if max_flex:
        v = decisive_bus(range(1, 15), base_net=base_net, kind=kind,
                         scale_factor=1 + flex)
        v['bus'] = v.index
        return v

    for f in np.linspace(-flex, flex, 5):
        v = decisive_bus(range(1, 15), base_net=base_net, kind=kind,
                         scale_factor=1 + f)
        v['bus'] = v.index
        if values is None:
            values = v
        else:
            values = pd.concat([values, v])
    return values

def nmbu_palette():
    """
    Createes list with rbg of nmbu colors. Can be fed to seaborn.set_palette()
    :return:
    """
    color_path = os.path.join(DATA_PATH,'nmbu_colors.xlsx')
    colorframe = pd.read_excel(color_path)
    colorframe = colorframe[['r', 'g', 'b']].values
    palette = [colorframe[k, :] for k in range(colorframe.shape[0])]
    return palette


def play_model(model_name='overnight_full'):

    for reward in ['voltage', 'current', 'imbalance']:
        length = 10000
        model, env = load_env(model_name,
                              seed=9)
        env.set_parameters({'reward_terms': [reward]})
        rewards, t_steps, hues, hours = [], [], [], []
        actions = []
        env._episode_start_hour = 0
        env._episode_start_day = 0
        env.solar_forecasts = env.get_episode_solar_forecast()
        env.demand_forecasts = env.get_episode_demand_forecast()
        obs = env.reset(reset_time=False)

        env2 = copy.deepcopy(env)
        env2.do_action = False
        sol = env.solar_forecasts[0]
        demand = env.demand_forecasts[0]

        show_sun, show_demand = True, True
        for t_step in range(1, length):

            action, _ = model.predict(obs)
            obs1, reward1, dones1, info1 = env.step(action)
            obs2, reward2, dones2, info2 = env2.step(action)

            current_step = env._current_step
            hour = calc_hour(env._episode_start_hour, current_step)

            if current_step == 0:
                sol = env.solar_forecasts[0]
                demand = env.demand_forecasts[0]

            rewards.append(reward1)
            hues.append('Agent')
            t_steps.append(t_step)
            hours.append(hour)

            rewards.append(reward2)
            hues.append('No agent')
            t_steps.append(t_step)
            hours.append(hour)

            rewards.append(action[0])
            hues.append('Action')
            t_steps.append(t_step)
            hours.append(hour)


            if show_sun:
                rewards.append(sol[env._current_step - 1])
                hues.append('Sun')
                t_steps.append(t_step)
                hours.append(hour)

            if show_demand:
                rewards.append(demand[env._current_step - 1])
                hues.append('Demand')
                t_steps.append(t_step)
                hours.append(hour)

        df = pd.DataFrame()
        df['Reward'] = rewards
        df['Hours'] = t_steps
        df['Hour in the day'] = hours
        df[''] = hues

        filename = '{}_hour_{}.csv'.format(model_name,reward)
        df.to_csv(os.path.join(DATA_PATH,'model_test',filename), index=False)


def store_actions(env, model, show_imbalance=False, show_solar=True,
                  show_action=True,
                  show_demand=False, period=10000):
    """
    A trained agent plays, and the actions are stored

    """
    net = env.powergrid
    actions, t_steps, flex_loads, sols, hours = [], [], [], [], []

    obs = env.reset()
    sol = env.solar_forecasts[0]
    demand = env.demand_forecasts[0]
    hues = []
    for t_step in range(1, period):

        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        if env._current_step == 0:
            sol = env.solar_forecasts[0]
            demand = env.demand_forecasts[0]

        if show_action:
            actions += list(action)
            hues += ['action' for _ in range(len(action))]
            t_steps += list(t_step * np.ones_like(action))
            flex_loads += list(net.load.index)

        if show_solar:
            actions += list(sol[env._current_step - 1] * np.ones_like(action))
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
            actions += list(demand[env._current_step - 1] * np.ones_like(action))
            hues += ['demand' for _ in range(len(action))]
            t_steps += list(t_step * np.ones_like(action))
            flex_loads += list(net.load.index)

        hour = calc_hour(env._episode_start_hour, env._current_step)
        hours += [hour for _ in range(len(action) * 2)]

    df = pd.DataFrame()
    df['actions'] = actions
    df['steps'] = t_steps
    #df['load'] = flex_loads
    df['hour'] = hours
    df[''] = hues
    return df

def plot_rewards_by_hour(df):
    """
    plots reward improvement between agent and no agent
    :param df: DataFrames from play_model
    """
    agent = df[df[''] == 'Agent']
    no_agent = df[df[''] == 'No agent']
    bad_normal = no_agent[no_agent['Reward'] < 0]
    bad_hours = bad_normal['Hours']
    bad_agent = agent[agent['Hours'].isin(bad_hours.values)]
    diff = pd.DataFrame()
    diff['Improvement'] = bad_agent['Reward'].values - bad_normal['Reward'].values
    diff['Hour'] = bad_agent['Hour in the day'].values
    zero_data = []
    for h in range(7, 24):
        if h not in set(diff['Hour']):
            zero_data.append({'Hour': h, 'Improvement': 0})

    if len(zero_data) > 0:
        diff = diff.append(zero_data)

    fig, axes = plt.subplots()
    sns.boxplot(x='Hour', y="Improvement", data=diff, ax=axes)
    axes.axhline(0, c=".5", ls='--')
    axes.set_xlim(2.5, 17.5)  # V: 7.4,22.5, I :(3.5,17.5)
    axes.set_xlabel('Hour of the day')
    axes.set_ylabel('Current reward improvement')
    plt.show()


if __name__ == '__main__':
    play_model(model_name='ppo_test_2')
    df = pd.read_csv(os.path.join(DATA_PATH,'model_test','ppo_test_2_hour_voltage.csv'))
    df = df.rename(columns={'Unnamed: 3': ''})



    plot_rewards_by_hour(df)
    model, env = load_env('ppo_test_2')
    store_actions(env,model)
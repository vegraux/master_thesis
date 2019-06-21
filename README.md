# rl_power
`rl_power` is a Python toolkit that applies
reinforcement learning algorithms in an electric power system. The project was
originally a master thesis written in partnership with Statnett,
the Norwegian Transmission system operator.

The agent is allowed to modify the power consumption at nodes in a
distribution system with high peak demand and high production from solar power.
The goal of the agent is to reduce the number of current and
voltage violations in the grid by increasing/ decreasing the consumption
at appropriate times. The increase/decrease in power consumption is intended
to be a simplified program for demand response that exploits the residential
flexibility in the grid.
[`pandapower`](http://www.pandapower.org/) is used
for power flow calculations and
[`stable-baselines`](https://github.com/hill-a/stable-baselines)
 for reinforcement learning.


## Installation
### Prerequisites
`rl_power` inherits the prerequisites from `stable-baselines` and `pandapower`.
 You need python >=3.5. You can find installation guides here:
- [stable-baseline installation](https://stable-baselines.readthedocs.io/en/master/guide/install.html#prerequisites)
- [pandapower installation](http://www.pandapower.org/start/)
## Motivation

## Code example
`ActiveEnv` is custom gym environment that can interact with reinforcement algorithms in `stable-baselines`. A PPO agent is trained in this example:

```python
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from gym_power.envs.active_network_env import ActiveEnv

env = ActiveEnv()
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000)
```

See more in the [notebook tutorials](https://github.com/vegraux/master_thesis/blob/master/tutorials/active_env_tutorial.ipynb)
The master thesis is can be found in the [solberg2019](https://github.com/vegraux/master_thesis/tree/master/tutorials)



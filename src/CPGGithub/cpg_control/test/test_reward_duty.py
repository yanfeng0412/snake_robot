import os

#os.chdir('/home/drl/PycharmProjects/rl_baselines/baselines')
import gym
import numpy as np

from my_envs.mujoco import *
from my_envs.base.command_generator import command_generator
from matplotlib.pylab import plt
from evaluate.plot_results import *

from utils.Logger import  IO

from utils.plot_reward_duty import *


env_name =   'CellrobotEnvCPG2-v0'  #
# for i in range(1,8):
#     rewards, commands , v_e=  plot_fitness_t(i,env_name = env_name,  save_plot_path='RewardDuty/reward_{}'.format(i), seed = 0)
#
#     IO('RewardDuty/fitness{}_param.pkl'.format(i)).to_pickle((rewards, commands))
#
#     plt.figure(figsize=(8,6))
#     plt.plot(rewards[:,0])
#     plt.savefig('RewardDuty/reward_curve{}.jpg'.format(i))

#plot_all_curve(1, env_name = 'CellrobotEnvCPG-v0', save_plot_path='RewardDuty/seed0_', seed = 0)

reward_choice = 9
rewards, commands, v_e=  plot_fitness_t(reward_choice,env_name = env_name, render=True,
                                        save_plot_path='RewardDuty/Testreward_{}'.format(reward_choice), seed = 0)
IO('RewardDuty/fitness{}_param.pkl'.format(reward_choice)).to_pickle((rewards, commands, v_e))

plt.figure(figsize=(15,6))
for i in range(rewards.shape[1]):
    plt.plot(rewards[:,i], label=str(i))
#plt.plot(rewards[:,1])
plt.legend()

plt.savefig('RewardDuty/reward_curve{}.jpg'.format(reward_choice))
plt.show()

vx = rewards[:,1]
vy = rewards[:,2]

plt.plot(vx)
plt.plot(vy)
plt.show()
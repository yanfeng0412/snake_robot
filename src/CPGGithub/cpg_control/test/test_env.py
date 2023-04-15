
import gym
import numpy as np

from my_envs.mujoco import *
from my_envs.base.command_generator import command_generator
from matplotlib.pylab import plt
def env_test():
    env = gym.make('CellrobotEnvCPG-v0')  # MyAnt-v2 Swimmer2-v2  SpaceInvaders-v0 CellrobotEnv-v0

    #env = gym.wrappers.Monitor(env, 'tmp/tmp.mp4', force=True)

    print('state: ', env.observation_space)
    print('action: ', env.action_space)
    q_dim = 1

    #command =  command_generator(10000, 0.01, 2, vx_range=(-0.2, 0.2), vy_range = (0,0), wyaw_range = (0,0))
    reward_fun = 1
    obs = env.reset()
    print('test')

    env.env

    max_step = 1000

    v_e = []
    c_command = []
    xyz = []

    rewards = []
    #action = np.ones(39) * (1)

    # sim_state.qpos[:] = qpos_int

    # sim.set_state(sim_state)
    # sim.forward()

    for i in range(max_step):
        env.render()
        action = env.action_space.sample()

        next_obs, reward, done, infos = env.step(action)
        obs = next_obs

        v_e.append(infos['velocity_base'])
        c_command.append(infos['commands'])
        xyz.append(infos['obs'][:3])
        rewards.append(infos['rewards'])

        # env.render(mode='rgb_array')#mode='rgb_array'
    env.close()
    dt =0.01


    v_e = np.array(v_e)



env_test()

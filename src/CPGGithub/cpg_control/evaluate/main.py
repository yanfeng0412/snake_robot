import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from baselines.common import plot_util as pu
from utils import IO

# 实验数据原始目录
group_dir = 'log-files/Mar_10_TRPO_Exp18'
# 实验环境
ENV_name = 'CellrobotEnvCPG2-v0'
# 实验序号
exp_id = 18
# 实验工程根目录
root_path = '/home/drl/PycharmProjects/rl_baselines/baselines'

exp_no_list= [ 1,   ]
#action_dim = 39


exp_path = os.path.join(root_path, group_dir, ENV_name )
exp_dir_list = os.listdir(exp_path)


results_dir = os.path.join(root_path,  group_dir , 'results')
monitor_dir = os.path.join(root_path,  group_dir ,  'monitor')

if not os.path.isdir(results_dir):
    print('create dir')
    os.makedirs(results_dir)

exp_list = IO(os.path.join(os.path.join(root_path, group_dir), 'exp_id{}_param.pkl').format(exp_id)).read_pickle()
exp_list_pd = pd.DataFrame(exp_list)

def find_ExpPath(n, exp_dir_list):
    exp_path = []
    for p in exp_dir_list:
        no_exp = int(p.split('-')[0].split('_')[1])  # aparse
        if  no_exp == n:
            exp_path.append(p)
    if len(exp_path) >1:
        raise Exception("The folder contains more than 2 path of EXP No.{}".format(n))
    if len(exp_path) == 0:
        raise Exception("Cannot find the path of EXP No.{}".format(n))
    return exp_path[0]
def plot_learning_curve(r, save_plot_path=None, exp_no=1):
    plt.figure()
    name = 'No.{} learning curve'.format(exp_no)
    plt.plot(r.progress.TimestepsSoFar, r.progress.EpRewMean)

    plt.title(name)
    if save_plot_path is not None:
        plt.savefig(os.path.join(save_plot_path, 'No_{}-learning_curve.jpg'.format(exp_no)))


def evaluate_fun(result_path, action_dim,   parms, exp_no =1):
    save_plot_path=os.path.abspath(os.path.join(results_dir,'No_{}-Curve'.format(exp_no)))
    reward_fun_choice = parms['reward_fun_choice']
    load_path = os.path.abspath(os.path.join(result_path, 'model/modelmodel'))
    alg = parms.alg
    seed = 0
    env = parms.env

    num_timesteps = 0
    network = 'mlp'
    save_path ='log'
    save_video_interval=0
    save_video_length=2000


    evaluate_path = os.path.join(result_path,'evaluate')
    os.makedirs(evaluate_path, exist_ok=True)

    evaluate_path = os.path.abspath(evaluate_path)

    os.environ["OPENAI_LOGDIR"] = evaluate_path
    print('OPENAI_LOGDIR = ', os.getenv('OPENAI_LOGDIR'))

    os.environ["REWARD_CHOICE"] = str(reward_fun_choice)
    print('REWARD_CHOICE = ', os.getenv('REWARD_CHOICE'))

    os.environ["ACTION_DIM"] = str(action_dim)
    print('ACTION_DIM = ', os.getenv('ACTION_DIM'))


    if parms['num_buffer'] is not None:
        os.environ["NUM_BUFFER"] = str(parms['num_buffer'])
        print('NUM_BUFFER = ', os.getenv('NUM_BUFFER'))

    if parms['command_mode'] is not None:
        os.environ["COMMAND_MODE"] = str(parms['command_mode'])
        print('COMMAND_MODE = ', os.getenv('COMMAND_MODE'))

    if parms['buffer_mode'] is not None:
        os.environ["BUFFER_MODE"] = str(parms['buffer_mode'])
        print('BUFFER_MODE = ', os.getenv('BUFFER_MODE'))

    os.system("python3 -m evaluate.run "  +
                  " --seed " + str(seed) +
                  " --env " + str(env) +
                  " --alg " + str(alg) +
                  " --num_timesteps " + str(num_timesteps) +
                  " --network " + str(network) +
                  " --save_path " + str(save_path) +
                  " --save_video_interval " + str(save_video_interval) +
                  " --save_video_length " + str(save_video_length) +
                  " --load_path " + str(load_path)+
                  " --play "+
                  " --save_plot_path " +str(save_plot_path)
                  )

results_list = list()
last_results = list()

for exp_no in exp_no_list:
    r_path = find_ExpPath(exp_no, exp_dir_list)
    result_path = os.path.join(exp_path, r_path)
    # read results and params

    results = pu.load_results(result_path)
    r = results[0]
    parms = exp_list_pd['exp{}'.format(exp_no)]

    action_dim = parms.action_dim

    # plot learning curve

    plot_learning_curve(r, results_dir,exp_no )

    # evaluate
    evaluate_fun(result_path, action_dim,  parms, exp_no )



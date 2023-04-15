from utils.instrument import VariantGenerator, variant, IO
import os
from datetime import datetime
import shutil
import glob
import paramiko
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class VG(VariantGenerator):

    @variant
    def env(self):
        return [ 'CellrobotEnvCPG2-v0']  #CellrobotEnvCPG-v0 'MyAnt-v2'
    @variant
    def alg(self):
        return ['trpo_mpi']  # trpo-mpi

    @variant
    def num_timesteps(self):
        return [1e7]

    @variant
    def network(self):
        return ['mlp']  #  mlp, cnn, lstm, cnn_lstm, conv

    @variant
    def action_dim(self):
        return [40,13 ]  # 12,13,39,40

    @variant
    def seed(self):
        return [0]

    @variant
    def reward_fun_choice(self):
        return [11]

    @variant
    def timesteps_per_batch(self):
        return [4096]

    @variant
    def gamma(self):
        return [0.9995]

    @variant
    def ent_coef(self):
        return [0.01]

    @variant
    def num_buffer(self):
        return [0,1]

    @variant
    def command_mode(self):
        return ['no']

    @variant
    def buffer_mode(self):
        return [1]
    @variant
    def CPG_enable(self):
        return [0]




exp_id = 20
EXP_NAME ='_TRPO'
group_note ="************ABOUT THIS EXPERIMENT****************\n" \
            " CPG2 test" \
            " "

log_group = None


n_cpu = 8
save_path ='log'
save_video_interval=0
save_video_length=200


# print choices
variants = VG().variants()
num=0
# for v in variants:
#     num +=1
#     print('exp{}: '.format(num), v)

# save gourp parms
exp_group_dir = datetime.now().strftime("%b_%d")+EXP_NAME+'_Exp{}'.format(exp_id)
group_dir = os.path.join('log-files', exp_group_dir)
if not os.path.exists(group_dir):
    os.makedirs(group_dir)  

variants = VG().variants()
num = 0
param_dict = {}
for v in variants:
    num += 1
    print('exp{}: '.format(num), v)
    parm = v
    parm = dict(parm, **v)
    param_d = {'exp{}'.format(num): parm}
    param_dict.update(param_d)


IO('log-files/' + exp_group_dir + '/exp_id{}_param.pkl'.format(exp_id)).to_pickle(param_dict)
print(' Parameters is saved : exp_id{}_param.pkl'.format(exp_id))
# save args prameters

# run
num_exp =0
for v in variants:
    num_exp += 1
    print(v)

    seed = v['seed']
    env  = v['env']
    network=v['network']
    alg = v['alg']
    num_timesteps =v['num_timesteps']



    log_group = exp_group_dir
    log_name = 'No_' + str(num_exp) + '-' + env + '_' + str(alg)
    if log_group is not None:
        root_path = os.path.join('log-files', log_group)
    else:
        root_path = os.path.join('log-files')
    now = datetime.now().strftime("%b-%d_%H:%M:%S")

    dir = os.path.join(root_path, env, log_name +"-" + now)
    dir = os.path.abspath(dir)

    os.environ["OPENAI_LOGDIR"] = dir
    print('OPENAI_LOGDIR = ', os.getenv('OPENAI_LOGDIR'))

    os.environ["REWARD_CHOICE"] = str(v['reward_fun_choice'])
    print('REWARD_CHOICE = ',os.getenv('REWARD_CHOICE'))

    if v['action_dim'] is not None:
        os.environ["ACTION_DIM"] = str(v['action_dim'])
        print('ACTION_DIM = ', os.getenv('ACTION_DIM'))


    if v['num_buffer'] is not None:
        os.environ["NUM_BUFFER"] = str(v['num_buffer'])
        print('NUM_BUFFER = ', os.getenv('NUM_BUFFER'))

    if v['command_mode'] is not None:
        os.environ["COMMAND_MODE"] = str(v['command_mode'])
        print('COMMAND_MODE = ', os.getenv('COMMAND_MODE'))

    if v['buffer_mode'] is not None:
        os.environ["BUFFER_MODE"] = str(v['buffer_mode'])
        print('BUFFER_MODE = ', os.getenv('BUFFER_MODE'))

    if v['CPG_enable'] is not None:
        os.environ["CPG_ENABLE"] = str(v['CPG_enable'])
        print('CPG_ENABLE = ', os.getenv('CPG_ENABLE'))
    # os.system("mpirun -np {} python3 -m baselines.run ".format(n_cpu)  +
    #           " --seed " + str(seed) +
    #           " --env " + str(env) +
    #
    #           " --alg " + str(alg) +
    #           " --num_timesteps " + str(num_timesteps) +
    #           " --network " + str(network) +
    #           " --save_path " + str(save_path) +
    #           " --save_video_interval " + str(save_video_interval) +
    #           " --save_video_length " + str(save_video_length)
    #
    #           )
    os.system("mpirun -np {} python3 -m baselines.run ".format(n_cpu) +
              " --seed " + str(seed) +
              " --env " + str(env) +

              " --alg " + str(alg) +
              " --num_timesteps " + str(num_timesteps) +
              " --network " + str(network) +
              " --save_path " + str(save_path) +
              " --save_video_interval " + str(save_video_interval) +
              " --save_video_length " + str(save_video_length)

              )
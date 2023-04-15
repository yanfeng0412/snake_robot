import os
import datetime

reward_choice = os.getenv('REWARD_CHOICE')

print(reward_choice)

exp_no = 1
env = 'CellRobot'
alg = 'TRPO'

EXP_name = 'TRPO'
log_name = 'No_' + str( exp_no) + '-' +  env   + '_' + str( alg)

log_group = None
if log_group is not None:
    root_path = os.path.join('log-files', log_group)
else:
    root_path = os.path.join('log-files' )
now = datetime.datetime.now().strftime("%b-%d_%H:%M:%S")

dir = os.path.join(root_path, log_name,  "-" + now)

os.environ["OPENAI_LOGDIR"]= dir
print(os.getenv('OPENAI_LOGDIR'))
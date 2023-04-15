# trpo_mpi

- Original paper: https://arxiv.org/abs/1502.05477
- Baselines blog post https://blog.openai.com/openai-baselines-ppo/
- `mpirun -np 16 python -m baselines.run --alg=trpo_mpi --env=PongNoFrameskip-v4` runs the algorithm for 40M frames = 10M timesteps on an Atari Pong. See help (`-h`) for more options.
- `python -m baselines.run --alg=trpo_mpi --env=Ant-v2 --num_timesteps=1e6` runs the algorithm for 1M timesteps on a Mujoco Ant environment. 
- also refer to the repo-wide [README.md](../../README.md#training-models)

python3 -m baselines.run --alg=trpo_mpi --env=Ant-v2 --num_timesteps=1e3 --save_path 'log-files/trpo/exp1'

!OPENAI_LOGDIR=$HOME/PycharmProjects/rl_baselines/baselines/log-files/logger OPENAI_LOG_FORMAT=csv python3 -m baselines.run --alg=trpo_mpi --env=Ant-v2 --num_timesteps=1e3 --save_path 'log-files/trpo/exp1'



python3 -m baselines.run --alg=trpo_mpi --env=CellrobotEnvCPG-v0 --num_timesteps=1e3
 
/tmp/openai-2019-03-01-19-48-26-192418
Logging to /tmp/openai-2019-03-01-19-48-26-192418



export REWARD_CHOICE=2

mpirun -np 8 python3 -m baselines.run --alg=trpo_mpi --env=CellrobotEnvCPG-v0 --num_timesteps=3e3  --network=mlp  --save_path='log' --save_video_interval=0
python3 -m baselines.run --alg=trpo_mpi --env=CellrobotEnvCPG-v0 --num_timesteps=3e3  --network=mlp --num_env=8 --save_path='log' --save_video_interval=0




python3 -m baselines.run --alg=trpo_mpi --env=CellrobotEnvCPG-v0 --num_timesteps=0 --load_path=~/models/pong_20M_ppo2 --play --save_path='log' --save_video_interval=1


export OPENAI_LOGDIR=/home/drl/PycharmProjects/rl_baselines/baselines/log-files/PPO2

python3 -m evaluate.run --alg=trpo_mpi --env=CellrobotEnvCPG-v0 --num_timesteps=0 --load_path=/home/drl/PycharmProjects/rl_baselines/baselines/log-files/Mar_04_TRPO_Exp1/CellrobotEnvCPG-v0/No_1-CellrobotEnvCPG-v0_trpo_mpi-Mar-04_23:18:04/model/modelmodel --play --save_path='log' --save_video_interval=1

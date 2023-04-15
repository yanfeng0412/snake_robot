import gym
from stable_baselines import PPO2 as ppo2
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
import balance_bot

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\projects\\python\\PPOC-balance-bot', 'D:/projects/python/PPOC-balance-bot'])

def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['log_interval'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return not is_solved

def main(mode="train"):

    n_cpu = 2
    env = SubprocVecEnv([lambda: gym.make('balancebot-v0') for i in range(n_cpu)])

    if mode == "train":
        model = ppo2(policy=MlpLnLstmPolicy,
                     env=env,
                     learning_rate=1e-3,
                     verbose=0,
                     full_tensorboard_log=True,
                     tensorboard_log="ppo2_balancebot_tensorboard")

        model.learn(
            total_timesteps=100000,
            callback=callback
        )
        print("Saving model to ppo2_balance.pkl")
        model.save("ppo2_balance.pkl")

        del model  # remove to demonstrate saving and loading

    if mode == "test":
        model = ppo2.load("ppo2_balance.pkl")

        obs = env.reset()
        done = [False, False]
        # env.set_done(5000)
        while not all(done):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            # env.render()
            print(obs)


if __name__ == '__main__':
    main(mode="train")
    exit(0)
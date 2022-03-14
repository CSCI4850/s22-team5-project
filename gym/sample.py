#!/usr/bin/env python3
import gym
from gym_doom.wrappers import ViZDoomEnv, ScreenWrapper

from stable_baselines3 import PPO

#env = gym.make("CartPole-v1")
#model = PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=10000)
#obs = env.reset()
#for i in range(1000):
#    action, _states = model.predict(obs, deterministic=True)
#    obs, reward, done, info = env.step(action)
#    env.render()
#    if done:
#      obs = env.reset()
#env.close()

if __name__ == '__main__':

    viz_doom_renderer = False
    dir_ = "E:\VizDOOM"
    env = gym.make('Doom-ram-v0')
    env = ViZDoomEnv(env, level='deadly_corridor', data_dir=dir_)
    env = ScreenWrapper(env, dim=(100, 100),  render=True, dummy_obs=True)

    env.playHuman()

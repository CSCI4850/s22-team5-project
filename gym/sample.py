#!/usr/bin/env python3
import gym
import vizdoomgym
import time
import sys

from stable_baselines3 import PPO

env = gym.make("VizdoomDefendCenter-v0")

if len(sys.argv) > 1:
    model = PPO.load(sys.argv[1])
else:
    model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="/tmp/sac/")
    model.learn(total_timesteps=1e4)

obs = env.reset()

model.save("models/"+str(time.time())+"_saved_model.zip")
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    time.sleep(1/60)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
env.close()


#!/usr/bin/env python3
import gym
import vizdoomgym
import time
import sys
import uuid
import numpy as np
import random

from stable_baselines3 import PPO

#VizdoomBasic-v0
#VizdoomCorridor-v0
#VizdoomDefendCenter-v0
#VizdoomDefendLine-v0
#VizdoomHealthGathering-v0
#VizdoomMyWayHome-v0
#VizdoomPredictPosition-v0
#VizdoomTakeCover-v0
#VizdoomDeathmatch-v0
#VizdoomHealthGatheringSupreme-v0

env = gym.make("VizdoomBasic-v0")
print("observation_space shape:",env.observation_space.shape)
print("action_space shape:",env.action_space.n)

def visualize(env,model,count):
    sum = 0
    obs = env.reset()
    action_counts=model.action_space.n*[0]
    for i in range(count):
        #save some states to train the autoencoder with
        if (random.randint(0,300)==0):
            with open("obs/"+str(uuid.uuid4()),"w") as file:
                np.savetxt(file,np.column_stack(obs),fmt='%1.1f')
        action, _states = model.predict(obs, deterministic=True)
        action_counts[action]+=1
        time.sleep(1/120)
        obs, reward, done, info = env.step(action)
        sum=sum+reward
        env.render()
        if done:
            obs = env.reset()
    print("average reward: "+str(sum))
    print(action_counts)

if len(sys.argv) > 1:
    model = PPO.load(sys.argv[1])
else:
    model = PPO("MlpPolicy", env, verbose=1)

identifier=uuid.uuid4()

print("training "+str(identifier))
while True:
    model.learn(total_timesteps=1e3)
    model.save("models/"+str(identifier)+"_saved_model.zip")
    simulation_reward_sum=visualize(env,model,600)
    print(simulation_reward_sum)

env.close()


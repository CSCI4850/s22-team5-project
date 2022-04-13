#!/usr/bin/env python3
import gym
from gym.utils.play import play
from gym.spaces import Box

import vizdoomgym
import time
import sys
import uuid
import numpy as np
import random
import os
import torch
import torchvision
import cv2
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import TensorDataset,DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

from stable_baselines3 import PPO


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.flatten=nn.Flatten()
        self.half=False
        self.encoder = nn.Sequential(
                nn.Linear(3*320*240, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
            )
        self.decoder = nn.Sequential(
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 3*320*240),
            )

    def forward(self,x):
   #     x = self.flatten(x)
        x = self.encoder(x)
        if self.half:
            return x
        y = self.decoder(x)
        return y

autoencoder=Autoencoder().to(device)
criterion=nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)

def autoencoder_error(img):
    img=torch.Tensor(img).to(device)
    img = img.reshape(-1,3*320*240)
    out = autoencoder(img)
    loss = criterion(out, img)
    return loss.item()

def train_autoencoder():
    X = []
    for root, dirs, files in os.walk('obs/'):
        for name in files:
            with open ("obs/"+name,"r") as f:
                X.append(np.loadtxt(f))
#plt.imshow(X[0],interpolation='nearest')
#plt.show()
    X=torch.Tensor(X).to(device)
    dataset = TensorDataset(X,X)
    loader=DataLoader(dataset)

    train_loss=[]
    outputs={}
    batch_size=len(loader)
    if batch_size < 1:
        return

    num_epochs=2
    #for epoch in tqdm(range(num_epochs)):
    for epoch in range(num_epochs):
        running_loss = 0
        for batch in loader:
            img, _ = batch
            img = img.reshape(-1,3*320*240)
            out = autoencoder(img)
            loss = criterion(out, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= batch_size
        train_loss.append(running_loss)
        outputs[epoch+1] = {'img': img, 'out': out}
#    print("running_loss: "+str(running_loss))

#torch.save(autoencoder.state_dict(),'autoencoder.pth')

class Autoencoder_wrapper(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.action_record=[0,0,0]
        self.reward_record=[0,0,0,0]

        _shape = (1,128)
        self.observation_space=Box(low=0,high=1,shape=_shape,dtype=np.float32)

    def step(self,action):
        self.action_record[action]+=1
        for i in range(3): # "look ma no frames"
            self.env.step(action)
        obs, reward, done, info = self.env.step(action)
        #save some states to train the autoencoder with
        if (random.randint(0,1000)==0):
            with open("obs/"+str(uuid.uuid4()),"w") as file:
                np.savetxt(file,np.column_stack(obs),fmt='%1.1f')
#        reward = reward + (autoencoder_error(np.column_stack(obs))/500)
        if reward == -1:
            self.reward_record[0]+=1
        if reward == -6:
            self.reward_record[1]+=1
        if reward == 100:
            self.reward_record[2]+=1
        if (random.randint(0,300)==0):
            print("action log:",self.action_record)
            print("reward log:",self.reward_record)
        img=torch.Tensor(obs).to(device)
        img = img.reshape(-1,3*320*240)
        autoencoder.half=True
        obs = autoencoder(img).cpu().detach()

        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        img=torch.Tensor(obs).to(device)
        img = img.reshape(-1,3*320*240)
        autoencoder.half=True
        ret = autoencoder(img)
        return ret.cpu().detach()

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
env = Autoencoder_wrapper(gym.make("VizdoomBasic-v0"))
print("observation_space shape:",env.observation_space.shape)
print("action_space shape:",env.action_space.n)

def visualize(env,model,count):
    sum = 0
    obs = env.reset()
    action_counts=model.action_space.n*[0]
    for i in range(count):
        action, _states = model.predict(obs, deterministic=True)
        action_counts[action]+=1
        time.sleep(1/120)
        obs, reward, done, info = env.step(action)
        sum=sum+reward
        env.render()
        if done:
            obs = env.reset()
    print("average reward: "+str(sum/count))
    print(action_counts)

if len(sys.argv) > 1:
    model = PPO.load(sys.argv[1])
else:
    model = PPO("MlpPolicy", env)

identifier=uuid.uuid4()

#train_autoencoder()
print("training "+str(identifier))
while True:
    model.learn(total_timesteps=1e3)
    model.save("models/"+str(identifier)+"_saved_model.zip")
    visualize(env,model,300)
#    train_autoencoder()

env.close()


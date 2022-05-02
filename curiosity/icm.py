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
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import TensorDataset,DataLoader

class Forward(nn.Module):
    def __init__(self,env):
        super(Autoencoder,self).__init__()
        self.flatten=nn.Flatten()
        self.model = nn.Sequential(
                nn.Linear(env.observation_space.size+env.action_space.n, 256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.ReLU(),
                nn.Linear(256, env.observation_space.size),
            )

    def forward(self,x):
        return self.model(x)

criterion=nn.MSELoss()
optimizer = optim.Adam(curiosity.parameters(), lr=1e-3, weight_decay=1e-5)

def curiosity_error(img):
    img=torch.Tensor(img).to(device)
    img = img.reshape(-1,3*screen_width*screen_height)
    out = curiosity(img)
    loss = criterion(out, img)
    return loss.item()

def train_curiosity(num_epochs=10):
    X=torch.Tensor(X).to(device)
    dataset = TensorDataset(X,X)
    loader=DataLoader(dataset,batch_size=8,shuffle=True)

    train_loss=[]
    batch_size=len(loader)
    if batch_size < 1:
        return

    i=0
    for epoch in range(num_epochs):
    #for epoch in tqdm(range(num_epochs)):
        running_loss = 0
        for batch in loader:
            img, _ = batch
            img = img.reshape(-1,3*screen_width*screen_height)
            out = curiosity(img)
            loss = criterion(out, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        i+=1
        running_loss /= batch_size
        train_loss.append(running_loss)
        print("running_loss: "+str(running_loss))
        if(i==1):
            i=0
            arr = np.reshape(out.cpu().detach().numpy()[0],(screen_height,screen_width,3))
            arr = np.round(arr).astype(int)
            arr = ndimage.rotate(arr,-90,reshape=True)
            arr = np.fliplr(arr)
            ax1.imshow(arr,interpolation='nearest')

            arr = np.reshape(img.cpu().detach().numpy()[0],(screen_height,screen_width,3))
            arr = np.round(arr).astype(int)
            arr = ndimage.rotate(arr,-90,reshape=True)
            arr = np.fliplr(arr)
            ax2.imshow(arr,interpolation='nearest')
            plt.pause(1/10)

class Curiosity_wrapper(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)

        _shape = (1,latent_width)
        self.observation_space=Box(low=0,high=1,shape=_shape,dtype=np.float32)

    def step(self,action):
        for i in range(4): # "look ma no frames"
            self.env.step(action)
        obs, reward, done, info = self.env.step(action)
        #save some states to train the curiosity with
        a_err = curiosity_error(np.column_stack(obs))
        reward = reward + (a_err/500)
        img=torch.Tensor(obs).to(device)
        img = img.reshape(-1,3*screen_width*screen_height)
        obs = curiosity(img).cpu().detach()

        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        img=torch.Tensor(obs).to(device)
        img = img.reshape(-1,3*screen_width*screen_height)
        ret = curiosity(img)
        return ret.cpu().detach()


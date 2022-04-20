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
                nn.Linear(env.observation_space.size+env.action_space.n, 128),
                nn.ReLU(),
                nn.Linear(128,128),
                nn.ReLU(),
                nn.Linear(128, env.observation_space.size),
            )

    def forward(self,x):
        return self.model(x)


#helps train the forward model, by keeping it focused
# on actions it can change
class Inverse(nn.Module):
    def __init__(self,env):
        super(Autoencoder,self).__init__()
        self.flatten=nn.Flatten()
        self.model = nn.Sequential(
                nn.Linear(env.observation_space.size*2, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, env.action_space.n),
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
    autoencoder.half=False
    img=torch.Tensor(img).to(device)
    img = img.reshape(-1,3*screen_width*screen_height)
    out = autoencoder(img)
    loss = criterion(out, img)
    return loss.item()

def train_autoencoder(num_epochs=10):
    print("training autoencoder")
    X = []
    i=0
    for root, dirs, files in os.walk('obs/'):
        for name in tqdm(files):
            with open ("obs/"+name,"r") as f:
                i+=1
                X.append(np.loadtxt(f))
    if (i == 0):
        return
#plt.imshow(X[0],interpolation='nearest')
#plt.show()
    X=torch.Tensor(X).to(device)
    dataset = TensorDataset(X,X)
    loader=DataLoader(dataset,batch_size=8,shuffle=True)

    train_loss=[]
    batch_size=len(loader)
    if batch_size < 1:
        return

    autoencoder.half=False
    i=0
    for epoch in range(num_epochs):
    #for epoch in tqdm(range(num_epochs)):
        running_loss = 0
        for batch in loader:
            img, _ = batch
            img = img.reshape(-1,3*screen_width*screen_height)
            out = autoencoder(img)
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

#torch.save(autoencoder.state_dict(),'autoencoder.pth')

class Autoencoder_wrapper(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)

        _shape = (1,latent_width)
        self.observation_space=Box(low=0,high=1,shape=_shape,dtype=np.float32)

    def step(self,action):
        for i in range(4): # "look ma no frames"
            self.env.step(action)
        obs, reward, done, info = self.env.step(action)
        #save some states to train the autoencoder with
        a_err = autoencoder_error(np.column_stack(obs))
        reward = reward + (a_err/500)
        if (random.randint(0,100)==0):
            print("a_err: ",a_err)
            if ((a_err > 5)):
                print("saving because a_err: ",a_err)
                with open("obs/"+str(uuid.uuid4()),"w") as file:
                    np.savetxt(file,np.column_stack(obs),fmt='%1.1f')
        img=torch.Tensor(obs).to(device)
        img = img.reshape(-1,3*screen_width*screen_height)
        autoencoder.half=True
        obs = autoencoder(img).cpu().detach()

        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        img=torch.Tensor(obs).to(device)
        img = img.reshape(-1,3*screen_width*screen_height)
        autoencoder.half=True
        ret = autoencoder(img)
        return ret.cpu().detach()


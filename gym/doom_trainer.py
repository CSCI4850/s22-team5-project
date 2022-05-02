#!/usr/bin/env python3
import gym
from gym.utils.play import play
from gym.spaces import Box

import random
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
import scipy.ndimage as ndimage
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import TensorDataset,DataLoader

#needed if we're running around like a horseman.
#import matplotlib
#matplotlib.use('Agg')

from matplotlib import pyplot as plt
from tqdm import tqdm

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

#env = gym.make("VizdoomMyWayHome-v0")
env = gym.make("VizdoomBasic-v0")
#env = gym.make("Pong-v0")

print("observation_space shape:",env.observation_space.shape)
print("action_space shape:",env.action_space.n)

latent_width=512

hl_width=512

screen_width=env.observation_space.shape[0]
screen_height=env.observation_space.shape[1]

print('screen_width: ',screen_width)
print('screen_height: ',screen_height)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.flatten=nn.Flatten()
        self.half=False
        self.encoder = nn.Sequential(
                nn.Linear(3*screen_width*screen_height, hl_width),
                nn.ReLU(),
                nn.Linear(hl_width, latent_width),
            )
        self.decoder = nn.Sequential(
                nn.Linear(latent_width, hl_width),
                nn.ReLU(),
                nn.Linear(hl_width, 3*screen_width*screen_height),
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

autoencoder_obs = []

figure, (ax1,ax2) = plt.subplots(2,1)
def train_autoencoder(num_epochs=10):
    print("training autoencoder with ",len(autoencoder_obs)," samples")
    X = autoencoder_obs
#    for root, dirs, files in os.walk('obs/'):
#        for name in tqdm(files):
#            with open ("obs/"+name,"r") as f:
#                i+=1
#                X.append(np.loadtxt(f))
    if (len(X) == 0):
        print("exiting train_autoencoder due to no samples")
        return
#plt.imshow(X[0],interpolation='nearest')
#plt.show()
    X=torch.Tensor(X).to(device)
    dataset = TensorDataset(X,X)
    loader=DataLoader(dataset,batch_size=8,shuffle=True)

    train_loss=[]
    batch_size=len(loader)
    if batch_size < 1:
        print("exiting train_autoencoder due to batch size < 1")
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
            arr = np.reshape(out.cpu().detach().numpy()[0],(screen_width,screen_height,3))
            arr = np.round(arr).astype(int)
            #arr = ndimage.rotate(arr,-90,reshape=True)
            #arr = np.fliplr(arr)
            ax1.imshow(arr,interpolation='nearest')

            arr = np.reshape(img.cpu().detach().numpy()[0],(screen_width,screen_height,3))
            arr = np.round(arr).astype(int)
            #arr = ndimage.rotate(arr,-90,reshape=True)
            #arr = np.fliplr(arr)
            ax2.imshow(arr,interpolation='nearest')
            plt.pause(1/10)

#torch.save(autoencoder.state_dict(),'autoencoder.pth')

class Autoencoder_wrapper(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)

        _shape = (1,latent_width)
        self.observation_space=Box(low=0,high=1,shape=_shape,dtype=np.float32)

    def step(self,action):
        obs, reward, done, info = self.env.step(action)
        #save some states to train the autoencoder with
        a_err = autoencoder_error(np.column_stack(obs))
        reward = reward + (a_err/500)
        if (random.randint(0,100)==0):
            print("a_err: ",a_err)
            if ((a_err > 400)):
                print("saving because a_err: ",a_err)
                if len(autoencoder_obs) > latent_width: #throw away a random observation
                    autoencoder_obs.pop(random.randrange(len(autoencoder_obs)))
                autoencoder_obs.append(obs)
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


class reward_wrapper(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.reward_sum=0
        self.reward_count=0
        self.observation_space=Box(low=0,high=1,shape=(75,50),dtype=np.float32)

    def downscale(self,obs):
        ret = cv2.resize(obs,dsize=(50,75))
        return ret[:,:,1]


    def step(self,action):
#        for i in range(2): # "look ma no frames"
#            self.env.step(action)
        obs,reward,done,info=self.env.step(action)
        if(done):
            print("reward_count: ",self.reward_count);
            print("reward_sum: ",self.reward_sum);
            self.reward_count=0
            self.reward_sum=0

        obs = self.downscale(obs)
        self.reward_count+=1
        self.reward_sum+=reward
        self.env.render()
        return obs,reward,done,info

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self.downscale(obs)

env = reward_wrapper(env)
#env = Autoencoder_wrapper(env)

#if len(sys.argv) > 1:
#    model = PPO.load(argv[1])
#else:
model = PPO("MlpPolicy", env,n_steps=2048,verbose=1,learning_rate=0.0001)

#model = PPO.load("actor.zip")
#model.exec()

identifier=uuid.uuid4()

should_train_autoencoder=True;
autoencoder_path='autoencoder.pth'
if should_train_autoencoder:
    train_autoencoder(num_epochs=20)
    torch.save(autoencoder.state_dict(),autoencoder_path)
else:
    autoencoder.load_state_dict(torch.load(autoencoder_path))
    autoencoder.eval()

#identifier=uuid.uuid4()
identifier="actor"
print("training "+str(identifier))
while True:
    model.learn(total_timesteps=1000)
#    print('saving model...')
#    model.save("actor.zip")
#    train_autoencoder(num_epochs=10)

env.close()


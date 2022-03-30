#!/usr/bin/env python3
import gym
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
from matplotlib import pyplot as plt
from tqdm import tqdm

from stable_baselines3 import PPO

#print("loading autoencoder...")
#ae = Autoencoder()
#ae.load_state_dict(torch.load('autoencoder.pth'))
#ae.eval()
#print("autoencoder loaded!")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Reward_wrapper(gym.RewardWrapper):
    def __init__(self,env):
        super().__init__(env)

    def reward(self,rew):
        return rew+autoencoder.running_loss

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
env = Reward_wrapper(gym.make("VizdoomBasic-v0"))
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
    visualize(env,model,600)

env.close()

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.flatten=nn.Flatten()
        self.encoder = nn.Sequential(
                nn.Linear(3*320*240, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 64),
            )
        self.decoder = nn.Sequential(
                nn.Linear(64, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 3*320*240),
            )

    def forward(self,x):
        x = self.flatten(x)
        x = self.encoder(x)
        y = self.decoder(x)
        return y

model=Autoencoder().to(device)

criterion=nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

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

num_epochs=200
for epoch in tqdm(range(num_epochs)):
    running_loss = 0
    for batch in loader:
        img, _ = batch
        img = img.reshape(-1,3*320*240)
        out = model(img)
        loss = criterion(out, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    running_loss /= batch_size
    train_loss.append(running_loss)
    outputs[epoch+1] = {'img': img, 'out': out}
#    print("running_loss: "+str(running_loss))

torch.save(model.state_dict(),'autoencoder.pth')

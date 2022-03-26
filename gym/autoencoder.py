#!/bin/env python3
import os
import torch
import torchvision
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import TensorDataset,DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
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

model=NeuralNetwork().to(device)

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
    # Iterating over the training dataset
    for batch in loader:
        img, _ = batch
        img = img.reshape(-1,3*320*240)
        out = model(img)
        # Calculating loss
        loss = criterion(out, img)
        # Updating weights according
        # to the calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Incrementing loss
        running_loss += loss.item()
    # Averaging out loss over entire batch
    running_loss /= batch_size
    train_loss.append(running_loss)
    # Storing useful images and
    # reconstructed outputs for the last batch
    outputs[epoch+1] = {'img': img, 'out': out}
#    print("running_loss: "+str(running_loss))

torch.save(model.state_dict(),'autoencoder.pth')

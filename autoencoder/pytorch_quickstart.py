#!/bin/env python3
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten=nn.Flatten()
        self.encoder = nn.Sequential(
                nn.Linear(3*320*280, 512),
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
                nn.Linear(512, 3*320*280),
            )

    def forward(self,x):
        x = self.flatten(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return logits

criterion=nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(num_epochs):
    running_loss = 0
    # Iterating over the training dataset
    for batch in train_loader:
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

transform=transforms.Compose([transforms.ToTensor()])
dataset = datasets.ImageFolder('images',transform=transform)

model = NeuralNetwork().to(device)
print(model)


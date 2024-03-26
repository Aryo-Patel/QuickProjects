import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP
import constants
import numpy as np
from tqdm import tqdm
from pyhessian import hessian

mnist_train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./data",
        download = True,
        train = True,
        transform = transforms.Compose([transforms.ToTensor()])
    ),
    batch_size = 10,
    shuffle=True,
)

device = constants.DEVICE
model = MLP().to(device)
loss_fn = nn.BCELoss()
optim = torch.optim.Adam(params=model.parameters())
epochs = 100
loss_landscape = []

def train_fn(data_loader, loss_fn, optim, model):
    loop = tqdm(data_loader)
    for i, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.to(device)
        optim.zero_grad()
        data = data.to(device)

        predictions = model(data)
        one_hot = F.one_hot(targets, num_classes = 10).float()
        # print(one_hot.dtype)
        # print(one_hot)
        # print(predictions.dtype)
        # print(predictions)
        loss = loss_fn(predictions, one_hot)
        loss.backward()
    
        optim.step()
        if i % 100 == 0:
            loop.set_postfix(loss=loss.item())

if __name__ == "__main__":
    for inputs, targets in mnist_train_loader:
        break
    one_hot = F.one_hot(targets, num_classes = 10).float()


    hessian_comp = hessian(model, loss_fn, data = (inputs, one_hot), mps=True)
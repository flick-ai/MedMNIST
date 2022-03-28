import numpy as np
import torch
from function import data_read
from template import Train, Test
from torch import nn
from Model import Resnet18
import torch.nn.functional as F

CUDA_on = True
cuda = CUDA_on and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
batch_size = 100
learning_rate = 0.0001
model = Resnet18().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
[train_data, val_data, test_data] = data_read("octmnist.npz", batch_size)

resnet18 = Train(train_data, val_data, device, model,
                 F.nll_loss, optimizer, 10, "./Net/Resnet18.pth")
resnet18.control()

test = Test(test_data, device, "./Net/Resnet18.pth")
test.test()

import numpy as np
from tqdm import tqdm
from function import count_parameters
from torch import nn
import torch
from torch.nn import functional as F


class Train:
    def __init__(self, data, val, device, model, loss, optimizer, time, path):
        self.device = device
        self.model = model
        self.data = data
        self.function = loss
        self.optimizer = optimizer
        self.path = path
        self.epoch = time
        self.val = val
        self.max = 0

    def control(self):
        print("Begin training:")
        print("The computing device:", "GPU" if self.device.type == "cuda" else "CPU")
        print("Total number of parameters:{}".format(str(count_parameters(self.model))))
        for i in range(self.epoch):
            self.train()

    def Save(self):
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(self.val), total=len(self.val)):
                self.model.eval()
                data, target = data.float(), target.long()
                data = data.resize(data.shape[0], 1, data.shape[1], data.shape[2])
                target = target.squeeze()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                prediction = output.data.max(1, keepdim=True)[1]
                correct += prediction.eq(target.data.view_as(prediction)).sum()
            correct = correct / len(self.val)
            print(correct)
            if correct > self.max:
                torch.save(self.model, self.path)

    def train(self):
        self.model.train()
        sum_loss = 0
        for batch_idx, (data, target) in tqdm(enumerate(self.data), total=len(self.data)):
            data, target = data.float(), target.long()
            data = data.resize(data.shape[0], 1, data.shape[1], data.shape[2])
            target = target.squeeze()
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data).float()
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            sum_loss += loss
        print(sum_loss / len(self.data))
        self.Save()
        return


class Test:
    def __init__(self, data, device, model):
        self.device = device
        self.model = torch.load(model)
        self.data = data

    def test(self):
        print("Begin testing:")
        print("The computing device:", "GPU" if self.device.type == "cuda" else "CPU")
        print("Total number of parameters:{}".format(str(count_parameters(self.model))))
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(self.data), total=len(self.data)):
                self.model.eval()
                data, target = data.float(), target.long()
                data = data.resize(data.shape[0], 1, data.shape[1], data.shape[2])
                target = target.squeeze()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                prediction = output.data.max(1, keepdim=True)[1]
                correct += prediction.eq(target.data.view_as(prediction)).sum()
            correct = correct / len(self.data)
            print(correct)

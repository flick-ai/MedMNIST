import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import cv2

Dataset = "D:/Pycharm/MedData/"


def data_read(name, batch_size):
    dataset = np.load(Dataset + name)
    train_images = torch.from_numpy(dataset['train_images'])
    train_labels = torch.from_numpy(dataset['train_labels'])
    val_images = torch.from_numpy(dataset['val_images'])
    val_labels = torch.from_numpy(dataset['val_labels'])
    test_images = torch.from_numpy(dataset['test_images'])
    test_labels = torch.from_numpy(dataset['test_labels'])
    train_dataset = TensorDataset(train_images, train_labels)
    train_data = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_images, val_labels)
    val_data = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_images, test_labels)
    test_data = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return [train_data, val_data, test_data]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


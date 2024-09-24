import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from functools import partial
import time
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
from pynvml import *
import opendatasets as od

class MultiLabelDataset(ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        multi_target = torch.zeros(len(self.classes))
        multi_target[target] = 1.
        
        return sample, multi_target 


def getdata(batch_size, multi_label=False, data_path="/home/cap6411.student1/CVsystem/assignment/hw5/human-action-recognition-dataset/Structured/"):

    isExist = os.path.exists(data_path)
    if isExist==False:
        dataset = 'https://www.kaggle.com/datasets/shashankrapolu/human-action-recognition-dataset/data'
        od.download(dataset)
    else:
        print("dataset exist")

    # Load the datasets
    data_dir =  data_path
    
    data_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ]) 

    # do data augmentation to get more train data
    data_transforms_train = transforms.Compose([
        transforms.RandomRotation(20),  # Randomly rotate the image within a range of (-20, 20) degrees
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with 50% probability
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Randomly crop the image and resize it
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly change the brightness, contrast, saturation, and hue
        transforms.RandomApply([transforms.RandomAffine(0, translate=(0.1, 0.1))], p=0.5),  # Randomly apply affine transformations with translation
        transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2)], p=0.5),  # Randomly apply perspective transformations
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Split out val dataset from train dataset
    if multi_label:
        train_dataset = MultiLabelDataset(data_dir+"train/", transform=data_transforms_train)
        test_dataset = MultiLabelDataset(data_dir+"test/", transform=data_transforms)
    else:
        train_dataset = datasets.ImageFolder(data_dir+"train/", transform=data_transforms_train)
        test_dataset = datasets.ImageFolder(data_dir+"test/", transform=data_transforms)
    n = len(train_dataset)
    n_val = int(0.1 * n)
    val_dataset = torch.utils.data.Subset(train_dataset, range(n_val))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    '''
    class_names = train_loader.dataset.classes
    for i in class_names:
        print(i)
    '''
    print("data load successfully!")
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, optimizer, loss_function, device, m_type):
    start_time = time.time()
    model.train()

    nvmlInit()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            if m_type == "siglip":
                outputs = model(inputs, labels)
                loss = outputs.loss
                loss_function.update(loss.item(), inputs.size(0))
                logits = outputs.logits
                sigmoid = torch.nn.Sigmoid()
                probs = sigmoid(logits.squeeze())
                max_ = probs.argmax (1)
                preds = nn.functional.one_hot(max_, num_classes=15)
            else:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        if m_type == "siglip":
            running_corrects += torch.sum((preds == labels.data).all(dim=1))
        else:
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print('Train Loss: {:.4f} Acc: {:.4f} Time: {:.0f}m {:.0f}s'.format(epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60)) # Modify this line
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'Train Epoch GPU memory used: {info.used/1000000:.4f} MB') 
    return epoch_loss, epoch_acc

def val_model(model, val_loader, loss_function, device, m_type):
    start_time = time.time()
    model.eval()

    nvmlInit()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

                
        with torch.no_grad():
            if m_type == "siglip":
                outputs = model(inputs, labels)
                loss = outputs.loss
                logits = outputs.logits
                sigmoid = torch.nn.Sigmoid()
                probs = sigmoid(logits.squeeze())
                max_ = probs.argmax (1)
                preds = nn.functional.one_hot(max_, num_classes=15)
            else:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_function(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        if m_type == "siglip":
            running_corrects += torch.sum((preds == labels.data).all(dim=1))
        else:
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print('Validation Loss: {:.4f} Acc: {:.4f} Time: {:.0f}m {:.0f}s'.format(epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60)) # Modify this line
    return epoch_loss, epoch_acc 

def test_model(model, test_loader, loss_function, device, m_type):
    start_time = time.time()

    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            if m_type == "siglip":
                outputs = model(inputs, labels)
                loss = outputs.loss
                logits = outputs.logits
                sigmoid = torch.nn.Sigmoid()
                probs = sigmoid(logits.squeeze())
                max_ = probs.argmax (1)
                preds = nn.functional.one_hot(max_, num_classes=15)
            else:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_function(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        if m_type == "siglip":
            running_corrects += torch.sum((preds == labels.data).all(dim=1))
        else:
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print('Test Loss: {:.4f} Acc: {:.4f} Time: {:.0f}m {:.0f}s'.format(epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60)) # Modify this line
    return epoch_loss, epoch_acc


if __name__ == '__main__':

    getdata(16, multi_label=True)


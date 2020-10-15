# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 12:06:01 2020

@author: Ania
"""

import matplotlib.pyplot as plt
import numpy as np
from SHBasis import SHBasis

import torch
import torch.nn as nn
from data_set_SHNet import load_data
from pytorch_lightning.metrics.regression import MSE

# Fully connected neural network 
class SHNet(nn.Module):
    def __init__(self, input_size):
        super(SHNet, self).__init__()
        self.bn0 = nn.BatchNorm1d(input_size)
        self.fc0 = nn.Linear(input_size, 150)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(150)
        self.fc = nn.Linear(150, 150)
        self.output = nn.Linear(150, input_size)

    def forward(self, x):
        out = self.bn0(x)
        out = self.fc0(out)
        out = self.relu(out)
        
        out = self.bn(out)
        out = self.fc(out)
        out = self.relu(out)
        
        out = self.bn(out)
        out = self.fc(out)
        out = self.relu(out)
        
        out = self.bn(out)
        out = self.output(out)
        return out


def train_model(model, criterion, optimizer_SGD, optimizer_Adam, metric, num_epochs, device):
    # Train the model
    model.train()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (SH_input, SH_label) in enumerate(train_loader):
            # Move tensors to the configured device
            SH_input = SH_input.to(device)
            SH_label = SH_label.to(device)

            # Forward pass
            outputs = model(SH_input)
            loss = criterion(outputs, SH_label)

            # Backward and optimize
            if epoch > 5:
                optimizer = optimizer_SGD
            else:
                optimizer = optimizer_Adam
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                         loss.item()))
            torch.save({
                        'epoch': epoch,
                        'mode_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss' : loss,
                        }, "model.pt"
            )

    # Validate the model
    model.eval()
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        mse = 0
        for SH_input, SH_label in test_loader:
            SH_input = SH_input.to(device)
            SH_label = SH_label.to(device)
            output = (model(SH_input))
            predicted = output.data
            mse += metric(predicted, SH_label)
            total += 1

        print('MSE of the network on the validation SH coefficient: {} %'.format(mse / total))
        return model


if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    input_size = 15
    num_epochs = 10
    batch_size = 128

    b_x = 1000
    b_y = 3000

    # data_dir = '/home/tomasz/data/brain_diffusion/hcp_wuminn/'
    data_dir = ''
    subject = '122317'
    filename_data = data_dir + subject + '/T1w/Diffusion/data.nii'
    filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
    filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'

    datasets = load_data(b_x, b_y, filename_data, filename_bvals, filename_bvecs)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=datasets["train"],
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=datasets["valid"],
                                              batch_size=batch_size,
                                              shuffle=False)

    model = SHNet(input_size).to(device)
    model.double()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer_Adam = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    metric = MSE()

    trained_model = train_model(model, criterion, optimizer_SGD, optimizer_Adam, metric, num_epochs, device)

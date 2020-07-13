import numpy as np
from collections import defaultdict
import torch
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_data
#from datasets import DHDataset
from DeepHarmony_NET import DeepHarmonyNet
import time

'''
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
'''


def train_model(model, optimizer,  num_epochs=200):

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
#                #inputs = inputs.to(device)
                #labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss =  torch.nn.L1Loss()
                    loss = loss(inputs, outputs)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            epoch_loss = loss / epoch_samples


        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Epoch loss :' + str(epoch_loss))


if __name__ == "__main__": 

    b_x = 1.0
    b_y = 3.0
    batch_size = 8
    max_epochs = 200

    dataloaders = load_data(b_x, b_y, batch_size)
    deepharmony_net = DeepHarmonyNet()
    optimizer = optim.Adam( deepharmony_net.parameters(), lr = 0.001)
    #summary(deepharmony_net, input_data=(1, 128, 128))
    model = train_model(deepharmony_net, optimizer,  num_epochs=max_epochs)


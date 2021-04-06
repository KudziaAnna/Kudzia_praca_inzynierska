import numpy as np
import torch
from torchsummary import summary
import torch.optim as optim
from datasets import load_data
from DeepHarmony_NET import DeepHarmonyNet
import time
import matplotlib.pyplot as plt
from SSIM import SSIM

'''
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
'''


def train_model(model, optimizer,  num_epochs=200):

    mae_loss =  torch.nn.L1Loss(reduction='mean')
    ssim_loss= SSIM()
    loss_train = []
    loss_val = []
    metric_train =[]
    metric_val =[]

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
            
            validation_pred = []
            validation_true = []
            validation_input = []

            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
#                #inputs = inputs.to(device)
                #labels = labels.to(device)
                
                
                # statistics
                epoch_samples += inputs.size(0)
                print(epoch_samples)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = mae_loss(inputs, outputs)
                    ssim = 1-ssim_loss(inputs, outputs)

                    if phase == 'val':
                        loss_val.append(loss.item())
                        metric_val.append(ssim.item())
                        outputs_np = outputs.numpy()
                        validation_pred.extend(
                            [outputs_np[s] for s in range(outputs_np.shape[0]) ]
                        )
                        labels_np = labels.numpy()
                        validation_true.extend(
                            [labels_np[s] for s in range(labels_np.shape[0]) ]
                        )
                        
                        inputs_np = inputs.numpy()
                        validation_input.extend(
                            [inputs_np[s] for s in range(inputs_np.shape[0]) ]
                        )

                    # backward + optimize only if in training phase
                    if phase == 'train':
    
                        loss_train.append(loss.item())
                        metric_train.append(ssim.item())
                        loss.backward()
                        optimizer.step()


        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Epoch loss :' + str(loss))
        print("Epoch SSIM "+str(ssim))

    print("Worst validation loss "+str( max(loss_val)))
    print("Best validation loss "+str( min(loss_val)))
    print("Worst validation SSIM metric"+str( min(metric_val)))
    print("Best validation SSIM metric "+str( max(metric_val)))
    return validation_pred, validation_true, validation_input

if __name__ == "__main__": 

    b_x = 3.0
    b_y = 1.0
    batch_size = 8
    max_epochs = 10

    dataloaders = load_data(b_x, b_y, batch_size)
    deepharmony_net = DeepHarmonyNet()
    optimizer = optim.Adam( deepharmony_net.parameters(), lr = 0.001)
    summary(deepharmony_net, input_data=(1, 96, 96))
    validation_pred, validation_true, validation_input = train_model(deepharmony_net, optimizer,  num_epochs=max_epochs)

    DH_error = np.abs( validation_true[1]- validation_pred[1])    
    b_error = np.abs( validation_true[1]- validation_input[1])    

    plt.subplot(3,3,5)
    plt.imshow(validation_pred[1].reshape([96,96]), cmap = 'gray')
    plt.axis('off')
    plt.title("DeepHarmony b-value = 1000")
    plt.subplot(3,3,2)
    plt.imshow(validation_true[1].reshape([96,96]), cmap = 'gray')
    plt.axis('off')
    plt.title("Original b-value=1000")
    plt.subplot(3,3,6)
    plt.imshow(DH_error.reshape([96,96]), cmap = 'gray')
    plt.axis('off')
    plt.title("Error after")
    plt.subplot(3,3,4)
    plt.imshow(b_error.reshape([96,96]), cmap = 'gray')
    plt.axis('off')
    plt.title("Error before")
    plt.subplot(3,3,8)
    plt.imshow(validation_input[1].reshape([96,96]), cmap = 'gray')
    plt.axis('off')
    plt.title("Original b-value=300")

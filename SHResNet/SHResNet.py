import torch
import torch.nn as nn
import numpy as np
from data_set_SHResNet import load_data

# Fully connected neural network
class SHResNet(nn.Module):
    def __init__(self, input_size, num_resBlock):
        super(SHResNet, self).__init__()
        self.cnn1 = nn.Conv3d(input_size, input_size, 3, padding=1)
        self.output = nn.Conv3d(input_size, 15, 3, padding=0)
        self.fu_0 = nn.Conv3d(1, 1, 3, padding=1)
        self.fu_2 = nn.Conv3d(5, 5, 3, padding=1)
        self.fu_4 = nn.Conv3d(9, 9, 3, padding=1)
        self.relu = nn.ReLU()
        self.num_resBlock = num_resBlock

    def forward(self, x):
        out = self.cnn1(x)
        tmp = out

        for i in range(self.num_resBlock):
            out = self.resBlock(out)
        out = self.cnn1(out)
        out = out - tmp
        out = self.cnn1(out)
        out = self.relu(out)

        out = self.cnn1(out)
        out = self.relu(out)

        out = self.output(out)
        out = self.relu(out)

        return out

    def func_unit(self, x, order):
        if order == 0:
            x = x[:, 0, :, :, :].reshape((x.shape[0], 1, x.shape[2], x.shape[3], -1))
            out = self.fu_0(x)
            out = self.relu(out)
            out = self.fu_0(out)
            out = self.relu(out)
            out = self.fu_0(out)
        elif order == 2:
            x = x[:, 1:6, :, :, :].reshape((x.shape[0], 5, x.shape[2], x.shape[3], -1))
            out = self.fu_2(x)
            out = self.relu(out)
            out = self.fu_2(out)
            out = self.relu(out)
            out = self.fu_2(out)
        else:
            x = x[:, 6:, :, :, :].reshape((x.shape[0], 9, x.shape[2], x.shape[3], -1))
            out = self.fu_4(x)
            out = self.relu(out)
            out = self.fu_4(out)
            out = self.relu(out)
            out = self.fu_4(out)

        return out

    def resBlock(self, x):
        fu0 = self.func_unit(x, 0)
        fu2 = self.func_unit(x, 2)
        fu4 = self.func_unit(x, 4)
        out = torch.cat([fu0, fu2, fu4], dim=1)
        return out-x

def train_model(model, criterion, optimizer_SGD, optimizer_Adam, num_epochs, device):
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
                'loss': loss,
            }, "model_SHResNet.pt"
            )

    # Validate the model
    model.eval()
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        total = 0
        mse = 0
        for SH_input, SH_label in test_loader:
            SH_input = SH_input.to(device)
            SH_label = SH_label.to(device)
            output = (model(SH_input))
            predicted = output.data
            mse += torch.mean((predicted - SH_label) ** 2)
            total += 1

        print('MSE of the network on the validation SH coefficient: {} %'.format(mse / total))


if __name__ == '__main__':
    # Device configuration
    device = torch.device( 'cpu')
    # 'cuda' if torch.cuda.is_available() else
    # Hyper-parameters
    num_resBlock = 2
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

    datasets = load_data(b_x, b_y)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=datasets["train"],
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=datasets["valid"],
                                              batch_size=batch_size,
                                              shuffle=False)

    model = SHResNet(input_size, num_resBlock).to(device)
    model.double()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer_Adam = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_model(model, criterion, optimizer_SGD, optimizer_Adam, num_epochs, device)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from data_set_SHResNet_cross import get_data

class SHResNet(nn.Module):
    def __init__(self, input_size, num_resBlock):
        super(SHResNet, self).__init__()
        self.cnn1 = nn.Conv3d(input_size, input_size, 3, padding=1)
        self.output = nn.Conv3d(input_size, input_size, 3, padding=0)
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

        out = self.cnn1(out)
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
        return out - x


def train_model(model, X, y, splits, criterion, optimizer_SGD, optimizer_Adam, num_epochs, device, batch_size):

    kfold = KFold(n_splits=splits, shuffle=True, random_state = 42)
    loss_data = []
    loss_test = []
    mse_test = []
    for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):
        print("Fold: " + str(fold + 1))

        X_train_fold = torch.tensor(X[train_index])
        y_train_fold = torch.tensor(y[train_index])
        X_test_fold = torch.tensor(X[test_index])
        y_test_fold = torch.tensor(y[test_index])

        train = torch.utils.data.TensorDataset(X_train_fold, y_train_fold)
        test = torch.utils.data.TensorDataset(X_test_fold, y_test_fold)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        
        for epoch in range(num_epochs):
            # Train the model
            model.train()
            total_train_step = len(train_loader)
            running_loss = 0
            
            for batch_idx, (SH_input, SH_label) in enumerate(train_loader):
                # Move tensors to the configured device
                SH_input = SH_input.to(device)
                SH_label = SH_label.to(device)

                # Forward pass
                outputs = model(SH_input)
                loss = criterion(outputs, SH_label)

                # Backward and optimize
                if epoch > 4:
                    optimizer = optimizer_SGD
                else:
                    optimizer = optimizer_Adam
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if (batch_idx + 1) % 2 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, num_epochs, batch_idx + 1,
                                                                                   total_train_step, running_loss))

            loss_data.append(running_loss / total_train_step)

            if (epoch +1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'mode_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, "/home/kudzia/SHResNet/models/SHResNet_cross_1_2_" + str((epoch + 1) + fold * num_epochs) + ".pt"
                )
                
            # Validate the model
            model.eval()
            total_test_step = len(test_loader)
            test_loss = 0
            mse = 0
            
            with torch.no_grad():
                total = 0
                for batch_idx, (SH_input, SH_label) in enumerate(test_loader):
                    SH_input = SH_input.to(device)
                    SH_label = SH_label.to(device)
                    output = (model(SH_input))
                    predicted = output.data
                    test_loss += loss.item()
                    mse += torch.mean((predicted - SH_label) ** 2)
                    total += 1

                    if (batch_idx + 1) % 1 == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Test Loss: {:.8f}, MSE: {:.8f}'.format(epoch + 1, num_epochs,
                                                                                      batch_idx + 1, total_test_step,
                                                                                      test_loss, mse))
                mse_test.append((mse / total_test_step))
                loss_test.append((test_loss / total_test_step))

    epoch_list = np.arange(1, splits * num_epochs + 1)
    plt.figure(0)
    plt.semilogy(epoch_list, loss_data, label="Train loss")
    plt.semilogy(epoch_list, loss_test, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("/home/kudzia/results/loss_cross_for_epoch_SHResNet.svg")

    plt.figure(1)
    plt.semilogy(epoch_list, mse_test)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.savefig("/home/kudzia/results/mseloss_cross_for_epoch_SHResNet.svg")

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # Hyper-parameters
    num_resBlock = 2
    input_size = 15
    num_epochs = 10
    batch_size = 128
    n_splits = 10

    b_x = 1000
    b_y = 2000
    # b_y = b_x

    X, y = get_data(b_x, b_y)

    model = SHResNet(input_size, num_resBlock).to(device)
    model.double()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer_Adam = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_model(model, X, y, n_splits, criterion, optimizer_SGD, optimizer_Adam, num_epochs, device, batch_size)

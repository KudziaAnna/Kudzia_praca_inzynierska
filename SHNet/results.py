import torch
import matplotlib.pyplot as plt
import numpy as np
from SHBasis import SHBasis
from SHNet import SHNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

b_x = 1000
b_y = 3000

# data_dir = '/home/tomasz/data/brain_diffusion/hcp_wuminn/'
data_dir = ''
subject = '122317'
filename_data = data_dir + subject + '/T1w/Diffusion/data.nii'
filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'

true_1000 = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order=4, bvalue=1000)
true_3000 = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order=4, bvalue=3000)

data_1000 = true_1000.get_SHCoeff()
data_3000 = true_3000.get_SHCoeff()

max_3000 = np.max(data_3000)
max_1000 = np.max(data_1000)
data_1000 = data_1000/max_1000
data_1000 = np.transpose(data_1000.reshape((data_1000.shape[0], -1)))
data_1000 = torch.from_numpy(data_1000).to(device)

model = SHNet(input_size=15).to(device)
model.double()
checkpoint = torch.load("model.pt")
model.load_state_dict(checkpoint['mode_state_dict'])
model.eval()

with torch.no_grad():
    harm_SHCoeff_1000 = model(data_1000)

harm_SHCoeff_1000 = max_3000 * harm_SHCoeff_1000.reshape((145, 174, 145, 15))
harm_SHCoeff_1000 = np.transpose(harm_SHCoeff_1000, (3, 0, 1, 2))

harm_3000 = true_3000.get_SHdata_from_SHCoeff(harm_SHCoeff_1000)
true_1000 = true_1000.data
true_3000 = true_3000.data


true_error = np.power((true_1000[:, :, :, 10] - true_3000[:, :, :, 10]), 2)
harm_error = np.power((true_3000[:, :, :, 10] - harm_3000[:, :, :, 10]), 2)

print("MSE before: "+ str(np.mean(true_error)))
print("MSE after: "+ str(np.mean(harm_error)))

s_axial = 70

plt.subplot(1, 5, 1)
plt.imshow(np.rot90(true_1000[:, :, s_axial, 12], 1), cmap='gray')
plt.axis("off")
plt.title("b = 1000")
plt.subplot(1, 5, 2)
plt.imshow(np.rot90(true_3000[:, :, s_axial, 12], 1), cmap='gray')
plt.axis("off")
plt.title("b = 3000")
plt.subplot(1, 5, 3)
plt.imshow(np.rot90(true_error[:, :, s_axial], 1), cmap='gray')
plt.axis("off")
plt.title("error before")
plt.subplot(1, 5, 4)
plt.imshow(np.rot90(harm_3000[:, :, s_axial, 12], 1), cmap='gray')
plt.axis("off")
plt.title("harm from 1 to 3")
plt.subplot(1, 5, 5)
plt.imshow(np.rot90(harm_error[:, :, s_axial], 1), cmap='gray')
plt.axis("off")
plt.title("after error")
plt.show()
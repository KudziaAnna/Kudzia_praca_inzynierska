import numpy as np
from torch.utils.data import Dataset
from SHBasis import SHBasis
from skimage.transform import resize


class SHResNetDataset(Dataset):
    """Brain dMRI dataset for harmonization"""

    def __init__(self, b_x, b_y):
        """Initialization"""
        l_order = 4

        data_dir = 'data/'
        subject = ['125525', '200109', '599671', '660951', '859671', '122317']
        for i in range(len(subject)):
            filename_data = data_dir + subject[i] + '/T1w/Diffusion/data.nii'
            filename_bvals = data_dir + subject[i] + '/T1w/Diffusion/bvals'
            filename_bvecs = data_dir + subject[i] + '/T1w/Diffusion/bvecs'
            sh_first = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order, b_x)
            sh_second = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order, b_y)
            if i == 0:
                sh_coeff_in = sh_first.get_SHCoeff()[:, 5, :5, :5]
                for i in range(1, sh_coeff_in.shape[1] - 1):
                    for j in range(1, sh_coeff_in.shape[2] - 1):
                        for k in range(1, sh_coeff_in.shape[3] - 1):
                            if i == 1 and j == 1 and k == 1:
                                input = sh_coeff_in[:, :3, :3, :3].reshape((15, 3, 3, 3, 1))
                            else:
                                tmp = sh_coeff_in[:, i - 1:i + 2, j - 1:j + 2, k - 1:k + 2].reshape((15, 3, 3, 3, 1))
                                input = np.append(input, tmp, axis=4)

                self.input = np.transpose(input, (4, 0, 1, 2, 3))
                print(self.input.shape)

                label = sh_second.get_SHCoeff()[:, 5, :5, :5]
                label = label[:, 1:sh_coeff_in.shape[1] - 1, 1:sh_coeff_in.shape[2] - 1, 1:sh_coeff_in.shape[3] - 1]
                print(label.shape)
                label = label.reshape((label.shape[0], 1, 1, 1, -1))
                self.label = np.transpose(label, (4, 0, 1, 2, 3))
                print(self.label.shape)

            else:
                sh_coeff_in = sh_first.get_SHCoeff()[:, 5, :5, :5]
                for i in range(1, sh_coeff_in.shape[1] - 1):
                    for j in range(1, sh_coeff_in.shape[2] - 1):
                        for k in range(1, sh_coeff_in.shape[3] - 1):
                            if i == 1 and j == 1 and k == 1:
                                input = sh_coeff_in[:, :3, :3, :3].reshape((15, 3, 3, 3, 1))
                            else:
                                tmp = sh_coeff_in[:, i - 1:i + 2, j - 1:j + 2, k - 1:k + 2].reshape((15, 3, 3, 3, 1))
                                input = np.append(input, tmp, axis=4)

                input = np.transpose(input, (4, 0, 1, 2, 3))
                self.input = np.concatenate((self.input, input), axis=0)
                print(self.input.shape)

                label = sh_second.get_SHCoeff()[:, 5, :5, :5]
                label = label[:, 1:sh_coeff_in.shape[1] - 1, 1:sh_coeff_in.shape[2] - 1, 1:sh_coeff_in.shape[3] - 1]
                label = label.reshape((label.shape[0], 1, 1, 1, -1))
                label = np.transpose(label, (4, 0, 1, 2, 3))
                self.label = np.concatenate((self.label, label), axis=0)
                print(self.label.shape)


        self.label = self.input
        if len(self.input) > len(self.label):
            self.input = self.input[0: len(self.label)]
        else:
            self.label = self.label[0: len(self.input)]

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.input)

    def __getitem__(self, index):
        """Generates one sample of data"""
        sh_input = self.input[index]
        sh_label = self.label[index]

        return [sh_input, sh_label]


def load_data(b_x, b_y):
    data = SHResNetDataset(b_x, b_y)

    indices = np.random.permutation(len(data))

    train_set = data[indices[:int(len(data) * 0.8)]]
    val_set = data[indices[int(len(data) * 0.8):]]

    datasets = {
        'train': train_set, 'valid': val_set
    }
    return datasets

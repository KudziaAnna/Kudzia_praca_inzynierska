import torch
import numpy as np
from torch.utils.data import Dataset
from SHBasis import SHBasis
from skimage.transform import resize


class SHResNetDataset(Dataset):
    """Brain dMRI dataset for harmonization"""

    def __init__(self, b_x, b_y):
        """Initialization"""
        l_order = 4

        data_dir = '/home/kudzia/data/'
        subject = ['125525', '200109', '599671', '660951', '859671']
        for i in range(len(subject)):
            print("jestem")
            filename_data = data_dir + subject[i] + '/T1w/Diffusion/data.nii'
            filename_bvals = data_dir + subject[i] + '/T1w/Diffusion/bvals'
            filename_bvecs = data_dir + subject[i] + '/T1w/Diffusion/bvecs'
            sh_first = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order, b_x)
            sh_second = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order, b_y)
            if i == 0:
                sh_coeff_in = sh_first.get_SHCoeff()
                sh_coeff_out = sh_second.get_SHCoeff()
                for i in range(1, sh_coeff_in.shape[1] - 1):
                    if i == 1:
                        input = sh_coeff_in[:, :2, :, :].reshape((1, sh_coeff_in.shape[0], 2, sh_coeff_in.shape[2], sh_coeff_in.shape[3]))
                        label = sh_coeff_out[:, :2, :, :].reshape((1, sh_coeff_out.shape[0], 2, sh_coeff_out.shape[2], sh_coeff_out.shape[3]))
                    else:
                        tmp_i = sh_coeff_in[:, i - 1:i + 1, :, :].reshape((1, sh_coeff_in.shape[0], 2, sh_coeff_in.shape[2], sh_coeff_in.shape[3]))
                        tmp_o = sh_coeff_out[:, i - 1:i + 1, :, :].reshape((1, sh_coeff_out.shape[0], 2, 
                                sh_coeff_out.shape[2], sh_coeff_out.shape[3]))
                        input = np.append(input, tmp_i, axis=0)
                        label = np.append(label, tmp_o, axis=0)
        
                self.input = input
                #self.input = np.transpose(sh_coeff_in, (1, 0, 2, 3))
                print(self.input.shape)
                '''
                label = sh_second.get_SHCoeff()
                
                label = label[:, 1:sh_coeff_in.shape[1] - 1, 1:sh_coeff_in.shape[2] - 1, 1:sh_coeff_in.shape[3] - 1]
                label = label.reshape((label.shape[0], 1, 1, 1, -1))
                
                self.label = label.reshape(1,label.shape[0], label.shape[1], label.shape[2], label.shape[3])
                '''
                self.label = label
                print(self.label.shape)

            else:
                sh_coeff_in = sh_first.get_SHCoeff()
                sh_coeff_out = sh_second.get_SHCoeff()
                for i in range(1, sh_coeff_in.shape[1] - 1):
                    if i == 1:
                        input = sh_coeff_in[:, :2, :, :].reshape((1, sh_coeff_in.shape[0], 2, sh_coeff_in.shape[2], sh_coeff_in.shape[3]))
                        label = sh_coeff_out[:, :2, :, :].reshape((1, sh_coeff_out.shape[0], 2, sh_coeff_out.shape[2], sh_coeff_out.shape[3]))
                    else:
                        tmp_i = sh_coeff_in[:, i - 1:i + 1, :, :].reshape((1, sh_coeff_in.shape[0], 2, sh_coeff_in.shape[2], sh_coeff_in.shape[3]))
                        tmp_o = sh_coeff_out[:, i - 1:i + 1, :, :].reshape((1, sh_coeff_out.shape[0], 2, 
                                sh_coeff_out.shape[2], sh_coeff_out.shape[3]))
                        input = np.append(input, tmp_i, axis=0)
                        label = np.append(label, tmp_o, axis=0)
                '''
                input = sh_coeff_in_next.reshape(1,sh_coeff_in_next.shape[0], sh_coeff_in_next.shape[1],
                           sh_coeff_in_next.shape[2], sh_coeff_in_next.shape[3])
                '''
                self.input = np.concatenate((self.input, input), axis=0)
                print(self.input.shape)
                '''
                label = sh_second.get_SHCoeff()
                label = label[:, 1:sh_coeff_in.shape[1] - 1, 1:sh_coeff_in.shape[2] - 1, 1:sh_coeff_in.shape[3] - 1]
                label = label.reshape((label.shape[0], 1, 1, 1, -1))
                
                label = label.reshape(1,label.shape[0], label.shape[1], label.shape[2], label.shape[3])
                '''
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
    train_set, val_set = torch.utils.data.random_split(data, [int(len(data)*0.9)+1, int(len(data)*0.1)])
    datasets = {
        'train': train_set, 'valid': val_set
    }
    return datasets

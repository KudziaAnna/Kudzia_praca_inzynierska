# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:48:31 2020

@author: Ania
"""

import numpy as np
from torch.utils.data import Dataset
from SHBasis import SHBasis


class SHNetDataset(Dataset):
    """Brain dMRI dataset for harmonization"""

    def __init__(self, b_x, b_y):
        """Initialization"""
        l_order = 4
        data_dir = '/home/kudzia/data/'
        subject = '122317'
        subject = ['125525', '200109', '599671', '660951', '859671']
        for i in range(len(subject)):
            filename_data = data_dir + subject[i] + '/T1w/Diffusion/data.nii'
            filename_bvals = data_dir + subject[i] + '/T1w/Diffusion/bvals'
            filename_bvecs = data_dir + subject[i] + '/T1w/Diffusion/bvecs'
            sh_first = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order, b_x)
            sh_second = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order, b_y)
            if i == 0:
                input = sh_first.get_SHCoeff()
                input = np.transpose(input.reshape((input.shape[0], -1)))
                label = sh_second.get_SHCoeff()
                label = np.transpose(label.reshape((label.shape[0], -1)))
            else:
                tmp_in = sh_first.get_SHCoeff()
                tmp_in = np.transpose(tmp_in.reshape((tmp_in.shape[0], -1)))
                input = np.concatenate((input, tmp_in), axis = 0)
                tmp_lab = sh_second.get_SHCoeff()
                tmp_lab = np.transpose(tmp_lab.reshape((tmp_lab.shape[0], -1)))
                label = np.concatenate((label, tmp_lab), axis = 0)

        self.input = input/np.max(input)
        print(self.input.shape)

        print(np.max(label))
        self.label = label/np.max(label)

    
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
        print(sh_input.shape)
        
        return [sh_input, sh_label]


def load_data(b_x, b_y):

    subject_size = 2204160
    data = SHNetDataset(b_x, b_y)

    test_set = data[(len(data)-subject_size):]
    
    indices = np.random.permutation(len(data))

    # train_set = data[indices[:int((len(data)-subject_size)*0.8)]]
    # val_set = data[indices[int((len(data)-subject_size)*0.8):(len(data)-subject_size)]]
    
    train_set = data[indices[:int(len(data)*0.8)]]
    val_set = data[indices[int(len(data)*0.8):]]
    
    datasets = {
        'train': train_set, 'valid': val_set, 'test': test_set
    }
    return datasets

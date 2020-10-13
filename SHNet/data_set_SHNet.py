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

    def __init__(self, b_x, b_y, filename_data, filename_bvals, filename_bvecs):
        """Initialization"""
        l_order = 4
        sh_first = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order, b_x)
        sh_second = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order, b_y)
        input = sh_first.get_SHCoeff()
        print(np.max(input))
        input = input/np.max(input)
        self.input = np.transpose(input.reshape((input.shape[0], input.shape[1]*input.shape[2]*input.shape[3])))
        label = sh_second.get_SHCoeff()
        print(np.max(label))
        label = label/np.max(label)
        self.label = np.transpose(label.reshape((label.shape[0], label.shape[1]*label.shape[2]*label.shape[3])))
    
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


def load_data(b_x, b_y, filename_data, filename_bvals, filename_bvecs):
    
    data = SHNetDataset(b_x, b_y, filename_data, filename_bvals, filename_bvecs)

    indices = np.random.permutation(len(data))

    train_set = data[indices[:int(len(data)*0.8)]]
    val_set = data[indices[int(len(data)*0.8):]]

    datasets = {
        'train': train_set, 'val': val_set
    }
    return datasets

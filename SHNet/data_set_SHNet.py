# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:48:31 2020

@author: Ania
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
from SHcoeff import SHcoeff

class SHNetDataset(Dataset):
    """Brain dMRI dataset for harmonization"""

    def __init__(self, b_x, b_y , filename_data, filename_bvals, filename_bvecs):
        'Initialization'
        l_order = 4
        SH_first = SHcoeff(filename_data, filename_bvals, filename_bvecs, l_order, b_x)
        SH_second = SHcoeff(filename_data, filename_bvals, filename_bvecs, l_order, b_y)
        self.input = SH_first.get_SHcoeff()
        self.label = SH_second.get_SHcoeff()
        if(len(self.input)> len(self.label)):
            self.input = self.input[0: len(self.label)]
        else:
            self.label = self.label[0: len(self.input)]


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input)

    def __getitem__(self, index):
        'Generates one sample of data'
        SH_input = self.input[index]
        SH_label = self.label[index]

        return [SH_input, SH_label]


def load_data(batch_size, b_x, b_y , filename_data, filename_bvals, filename_bvecs):
    
    data = SHNetDataset(b_x , b_y , filename_data, filename_bvals, filename_bvecs)

    indices = np.random.permutation(len(data))

    train_set = data[indices[:int(len(data)*0.8) ] ]
    val_set = data[indices[int(len(data)*0.8) : ] ]

    datasets = {
        'train': train_set, 'val': val_set
    }
    return datasets
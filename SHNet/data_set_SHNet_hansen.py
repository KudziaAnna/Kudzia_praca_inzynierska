# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:48:31 2020

@author: Ania
"""
import numpy as np
from SHBasis import SHBasis

def get_data( b_x, b_y):
    """Initialization"""
    l_order = 4
    data_dir = '/home/kudzia/data/hansen/'

    filename_data = data_dir + 'human_brain.mat'
    filename_bvals = data_dir + 'effective_bvalues.txt'
    filename_bvecs = data_dir + 'DWI_encoding_directions.txt'

    sh_first = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order, b_x)
    sh_second = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order, b_y)

    input = sh_first.get_SHCoeff()
    print(input.shape)
    input = np.transpose(input.reshape((input.shape[0], -1)))
    label = sh_second.get_SHCoeff()
    label = np.transpose(label.reshape((label.shape[0], -1)))

    if len(input) > len(label):
        input = input[0: len(label)]
    else:
        label = label[0: len(input)]

    return input, label


if __name__ == '__main__':
    X, y = get_data(1.0, 1.4)
    print(X.shape)
    
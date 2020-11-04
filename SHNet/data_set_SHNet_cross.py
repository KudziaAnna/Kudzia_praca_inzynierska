# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:48:31 2020

@author: Ania
"""
import torch
import numpy as np
from SHBasis import SHBasis


def get_data( b_x, b_y):
    """Initialization"""
    l_order = 4
    data_dir = '/home/kudzia/data/'
    # data_dir = 'data/'
    subject = ['111312', '125525', '149741', '172332', '200109', '599671', '660951', '859671', '917255', '122317']
    #  '660951', '859671', '917255', '122317'
    for i in range(len(subject)):
        filename_data = data_dir + subject[i] + '/T1w/Diffusion/data.nii'
        filename_bvals = data_dir + subject[i] + '/T1w/Diffusion/bvals'
        filename_bvecs = data_dir + subject[i] + '/T1w/Diffusion/bvecs'
        sh_first = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order, b_x, normalized=True)
        sh_second = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order, b_y, normalized=True)
        if i == 0:
            input = sh_first.get_SHCoeff()
            input = np.transpose(input.reshape((input.shape[0], -1)))
            label = sh_second.get_SHCoeff()
            label = np.transpose(label.reshape((label.shape[0], -1)))
        else:
            tmp_in = sh_first.get_SHCoeff()
            tmp_in = np.transpose(tmp_in.reshape((tmp_in.shape[0], -1)))
            input = np.concatenate((input, tmp_in), axis=0)
            tmp_lab = sh_second.get_SHCoeff()
            tmp_lab = np.transpose(tmp_lab.reshape((tmp_lab.shape[0], -1)))
            label = np.concatenate((label, tmp_lab), axis=0)


    if len(input) > len(label):
        input = input[0: len(label)]
    else:
        label = label[0: len(input)]

    return input, label



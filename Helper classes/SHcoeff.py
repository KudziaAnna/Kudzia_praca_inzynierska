# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:00:11 2020

@author: Ania

Class that calculates l_order RISH features for specific b_value data

_transformIntoSH - private method that transforms b_vals into spherical coordinates
get_SHcoeff - public method that returns l_order SH coefficient according to Descoutex
"""


import numpy as np
import matplotlib.pyplot as plt

# niftii support
import nibabel as nib

# DIPY library
from dipy.io import read_bvals_bvecs
from dipy.reconst.csdeconv import real_sym_sh_basis
from dipy.core.gradients import gradient_table
from dipy.core.sphere import cart2sphere

class SHcoeff:

    
    def __init__(self, filename_data, filename_bvals, filename_bvecs, l_order, bvalue):
        img = nib.load(filename_data)
        data_volume = img.get_fdata()
        bvals, bvecs = read_bvals_bvecs(filename_bvals, filename_bvecs)
        self.l_order = l_order
        
        sel_b = bvals == bvalue
        self.data = data_volume[..., sel_b]
        self.bvals_sel = bvals[sel_b]
        self.bvecs_sel = bvecs[sel_b]

    
    
    def _transformIntoSH(self):
         r, Phi, Theta = cart2sphere(self.bvecs_sel[:,0], self.bvecs_sel[:,1], self.bvecs_sel[:,2])
         return r, Phi, Theta
    
    
    def get_SHcoeff(self):
        r, Phi, Theta = self._transformIntoSH()
        SH = real_sym_sh_basis(self.l_order, Theta, Phi)[0]
        return SH
    

    
    
if __name__ == '__main__':
    subject = '122317'

    #data_dir = '/home/tomasz/data/brain_diffusion/hcp_wuminn/'
    data_dir = ''
    filename_data =  data_dir + subject + '/T1w/Diffusion/data.nii'
    filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
    filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'
    
    SH_class = SHcoeff(filename_data, filename_bvals, filename_bvecs, 4, 1000)
    SH = SH_class.get_SHcoeff()
   
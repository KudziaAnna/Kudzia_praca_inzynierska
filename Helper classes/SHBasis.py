# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:42:07 2020

@author: Ania

Class that calculates l_order RISH features for specific b_value data

_calculate_L - private method that calculates matrix L
_calculate_B - private method that calculates matrix B
get_SHCoeff - public method that calculates vector C of SH Coefficients
_calculate_L - private method that calculates matrix L
_transformIntoSH - private method that transforms b_vals into spherical coordinates
get_RISH - public method that returns l_order RISH features
get_SH - public method 

Markings in accordance with  https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21277

"""
import nilearn
import numpy as np
import matplotlib.pyplot as plt

# niftii support
import nibabel as nib

# DIPY library
from dipy.io import read_bvals_bvecs
from dipy.reconst.csdeconv import real_sym_sh_basis
from dipy.core.sphere import cart2sphere


class SHBasis:

    def __init__(self, filename_data='122317/T1w/Diffusion/data.nii', filename_bvals='122317/T1w/Diffusion/bvals', filename_bvecs='122317/T1w/Diffusion/bvecs', l_order=4, bvalue=3000):
        img = nib.load(filename_data)
        data_volume = img.get_fdata()
        bvals, bvecs = read_bvals_bvecs(filename_bvals, filename_bvecs)
        self.l_order = l_order
        
        sel_b = bvals == bvalue
        self.data = data_volume[..., sel_b]
        self.bvecs_sel = bvecs[sel_b]

        
    def _calculate_L(self):
        R = int((self.l_order+1)*(self.l_order+2)/2)
        L = np.zeros((R, R))
        R_tmp = 1
        l_j = 0
        for j in range(0, R):
            if j == R_tmp:
                l_j +=2
                R_tmp = int((l_j+1)*(l_j+2)/2)
        
            L[j, j] = np.power(l_j, 2)*np.power(l_j+1,  2)
        
        return L

    def _transformIntoSH(self):
         _, Phi, Theta = cart2sphere(self.bvecs_sel[:, 0], self.bvecs_sel[:, 1], self.bvecs_sel[:, 2])
         return Phi, Theta

    def _calculate_B(self):
        Phi, Theta = self._transformIntoSH()
        N = self.data.shape[3]
        for j in range(0, N):
            if j == 0:
                B = real_sym_sh_basis(self.l_order, Theta[j], Phi[j])[0]
            else:
                new = real_sym_sh_basis(self.l_order, Theta[j], Phi[j])[0]
                B = np.concatenate((B, new))
        
        return B

    def get_SHCoeff(self):
        par_lambda = 0.006
        B = self._calculate_B()
        L = self._calculate_L()
        S = np.transpose(self.data, (0, 1, 3, 2))
        B_s = np.dot(B.T, S)
        B_s = np.transpose(B_s, (1, 2, 0, 3))
        tmp = np.dot(B.T, B) + par_lambda*L
        B_l = np.linalg.inv(tmp)   
       
        C = np.dot(B_l, B_s)
        return C
    
    def get_RISH(self):
        C = self.get_SHCoeff()
        size = int(self.l_order/2 + 1)
        rish = [0]*size
        for i in range(0, size):
            for j in range(0, 4*i+1):
                rish[i] += np.power(C[j], 2)
                
        return rish
    
    def get_RISH_from_SHCoeff(self, C):
        size = int(self.l_order/2 + 1)
        rish = [0]*size
        for i in range(0, size):
            for j in range(0, 4*i+1):
                rish[i] += np.power(C[j], 2)
                
        return rish
    
    def get_SH(self):
        Phi, Theta = self._transformIntoSH()
        SH = real_sym_sh_basis(self.l_order, Theta, Phi)[0]
        return SH
    
    def get_SHdata(self):
        C = np.transpose(self.get_SHCoeff(), (1, 2, 0, 3))
        B = self._calculate_B()
        return np.transpose(np.dot(B, C), (1, 2, 3, 0))

    def get_SHdata_from_SHCoeff(self, C):
        """ C should be R x image size(a,b,c)"""
        C = np.transpose(C, (1, 2, 0, 3))
        B = self._calculate_B()
        print(B.shape)
        return np.transpose(np.dot(B, C), (1, 2, 3, 0))
    
    def error(self):
        return np.abs(self.data - self.get_SHdata())


if __name__ == '__main__':
    subject = '122317'

    # data_dir = '/home/tomasz/data/brain_diffusion/hcp_wuminn/'
    data_dir = ''
    filename_data = data_dir + subject + '/T1w/Diffusion/data.nii'
    filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
    filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'
    
    RISH_class = SHBasis(filename_data, filename_bvals, filename_bvecs, 8, 3000)
    RISH = RISH_class.get_RISH()
    SHCoeff = RISH_class.get_SHCoeff()

    plt.imshow(RISH_class.error()[:, :, 70, 5], cmap='gray')
    print(np.mean(RISH_class.error()))

    s_axial = 70

    plt.subplot(1, 5, 1)
    plt.imshow(np.rot90(RISH[0][:, :, s_axial], 1), cmap='gray')
    plt.axis("off") 
    plt.title("0 order")
    plt.subplot(1, 5, 2)
    plt.imshow(np.rot90(RISH[1][:, :, s_axial], 1), cmap='gray')
    plt.axis("off") 
    plt.title("2 order")
    plt.subplot(1, 5, 3)
    plt.imshow(np.rot90(RISH[2][:, :, s_axial], 1), cmap='gray')
    plt.axis("off")
    plt.title("4 order") 
    plt.subplot(1, 5, 4)
    plt.imshow(np.rot90(RISH[3][:, :, s_axial], 1), cmap='gray')
    plt.axis("off") 
    plt.title("6 order")
    plt.subplot(1, 5, 5)
    plt.imshow(np.rot90(RISH[4][:, :, s_axial], 1), cmap='gray')
    plt.axis("off") 
    plt.title("8 order")

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
import numpy as np

# niftii support
import nibabel as nib
import scipy.io as sio
import matplotlib.pyplot as plt

# DIPY library
from dipy.io import read_bvals_bvecs
from dipy.reconst.csdeconv import real_sym_sh_basis
from dipy.core.sphere import cart2sphere


class SHBasis:

    def __init__(self, filename_data='122317/T1w/Diffusion/data.nii', filename_bvals='122317/T1w/Diffusion/bvals', filename_bvecs='122317/T1w/Diffusion/bvecs', l_order=4, bvalue=3000, normalized = False):
        '''
        img = nib.load(filename_data)
        data_volume = img.get_fdata()
        '''
        mat_data = sio.loadmat(filename_data)
        data_volume = mat_data['human_brain_data']

        bvals, bvecs = read_bvals_bvecs(filename_bvals, filename_bvecs)
        self.l_order = l_order

        cut_x = (35, 110)
        cut_y = (35, 134)
        cut_z = (25, 100)

        sel_b = bvals == bvalue
 
        self.all_data = data_volume[cut_x[0]: cut_x[1], cut_y[0]: cut_y[1], cut_z[0]: cut_z[1], :]
        self.data = data_volume[cut_x[0]: cut_x[1], cut_y[0]: cut_y[1], cut_z[0]: cut_z[1], sel_b]

        sel_b = bvals == bvalue
 
        self.all_data = data_volume
        self.data = data_volume[:, :, :, sel_b]
        self.bvecs_sel = bvecs[sel_b]
        idx0 = np.where(bvals < 100)[0]
        S0 = np.mean(self.all_data[:, :, :, idx0], axis=3)
        data_b0 = np.copy(S0)
        data_b0 = data_b0[:, :, :, np.newaxis] 
        
        self.mask =  S0 > 2500 
         
        # print("Max S0: "+str(np.max(S0)))
        # self.mask =  S0 > 30                     
        
        if normalized:
          self.data = np.divide(self.data, np.repeat(data_b0, self.data.shape[3], axis=3))
    
          self.data[np.isnan(self.data)] = 0
          self.data[self.data> 1] = 1
          self.data[self.data < 1e-3] = 1e-3
    
        self.data = self.data * self.mask[:, :, :, np.newaxis]     
        
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
        C[C<0.001] = 0.001
        return C
    
    def get_RISH(self):
        C = self.get_SHCoeff()
        size = int(self.l_order/2 + 1)
        rish = [0]*size
        for i in range(0, size):
            for j in range(0, 4*i+1):
                rish[i] += np.power(C[j], 2)
                
        return rish
    
    def get_RISH_from_SHCoeff(self, C, order):
        size = int(self.l_order/2 + 1)
        rish = [0]*size
        for i in range(0, size):
            for j in range(0, 4*i+1):
                rish[i] += np.power(C[j], 2)
                
        return rish[int(order/2)]
    
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
    data_dir = '/home/kudzia/data/'

    filename_data = data_dir + subject + '/T1w/Diffusion/data.nii'
    filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
    filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'
    '''
    filename_data = data_dir + 'human_brain.mat'
    filename_bvals = data_dir + 'effective_bvalues.txt'
    filename_bvecs = data_dir + 'DWI_encoding_directions.txt'
    '''    
    RISH_class = SHBasis(filename_data, filename_bvals, filename_bvecs, 4, 1000, normalized=False)
    SH = RISH_class.get_SHCoeff()
    print(np.max(SH))
    print(np.min(SH))
    orig = RISH_class.data
    masked = orig*RISH_class.mask
  
    error_1 = np.abs(orig-masked)
    error_2 = np.abs(orig-masked)
    print("Error_1: "+str(np.mean(error_1)))
    print("Error_2: "+str(np.mean(error_2)))
    
    s_axial = 65

    plt.subplot(2, 3, 1)
    plt.imshow(np.rot90(orig[:, :, s_axial, 10], 1), cmap='gray')
    plt.axis("off") 
    plt.title("Original")
    plt.subplot(2, 3, 2)
    plt.imshow(np.rot90(masked[:, :, s_axial, 10], 1), cmap='gray')
    plt.axis("off") 
    plt.title("Orig with mask")
    plt.subplot(2, 3, 3)
    plt.imshow(np.rot90(error_1[:, :, s_axial, 10], 1), cmap='gray')
    plt.axis("off")
    plt.title("Error") 
    plt.subplot(2, 3, 4)
    plt.imshow(np.rot90(orig[:, :, s_axial, 10], 1), cmap='gray')
    plt.axis("off") 
    plt.title("Original")
    plt.subplot(2, 3, 5)
    plt.imshow(np.rot90(masked[:, :, s_axial, 10], 1), cmap='gray')
    plt.axis("off") 
    plt.title("Decomp. bet")
    plt.subplot(2, 3, 6)
    plt.imshow(np.rot90(error_2[:, :, s_axial, 0]*255, 1), cmap='gray')
    plt.axis("off") 
    plt.title("Error")
    plt.show()

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:16:38 2020

@author: Ania
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# niftii support
import nibabel as nib

# DIPY library
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import real_sym_sh_basis
from dipy.core.sphere import cart2sphere


l_order = 6
R = int((l_order+1)*(l_order+2)/2)
N = 16
par_lambda = 0.006

#%% Calculating vector  matrix L
L = np.zeros((R,R))
R_tmp = 1
l_j = 0


for j in range(0, R):
    if j == R_tmp:
        l_j +=2
        R_tmp = int((l_j+1)*(l_j+2)/2)
        

    L[j,j] = np.power(l_j,2)*np.power(l_j+1,2)  



#%% Data loading - HCP WuMinn
#subject = '599671'
subject = '122317'

#data_dir = '/home/tomasz/data/brain_diffusion/hcp_wuminn/'
data_dir = ''
filename_data =  data_dir + subject + '/T1w/Diffusion/data.nii'
filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'
filename_mask =  data_dir + subject + '/fsl_' + subject + '_data_brain_mask.mat'

# load HCP data
img = nib.load(filename_data)
data_volume = img.get_fdata()

# load b-values and vectors
bvals_original, bvecs_original = read_bvals_bvecs(filename_bvals, filename_bvecs)
gtab = gradient_table(bvals_original, bvecs_original)

# load binary mask
mask_dict = loadmat(filename_mask)
data_mask = mask_dict.get("mask")
#%% Selecting specific b values data

bvals = gtab.bvals
bvecs = gtab.bvecs

sel_b = bvals == 3000
data = data_volume[..., sel_b]
bvals_sel = bvals[sel_b]
bvecs_sel = bvecs[sel_b]

#%%Transformating gradient direction G into spherical coordinates
# theta 0- 2pi, phi 0-pi

r, Phi, Theta = cart2sphere(bvecs_sel[:,0], bvecs_sel[:,1], bvecs_sel[:,2])

#%% Creating matrix B

for j in range(0,N):
   if j == 0:
       B = real_sym_sh_basis(l_order, Theta[j], Phi[j])[0]
   else:
       new = real_sym_sh_basis(l_order, Theta[j], Phi[j])[0]
       B  = np.concatenate((B, new))

#%% Calculatirn vector C

S = np.transpose(data, (0, 1,3,2))
B_s =  np.dot(B.T, S)
B_s= np.transpose(B_s,(1,2,0,3))

B_l = np.linalg.inv( np.dot(B.T, B)+par_lambda*L)  

       
C = np.dot(B_l, B_s)

#%% RISH features calculation

RISH = [0, 0, 0, 0]
for i in range(0,4):
    for j in range(0, 4*i+1):
        RISH[i] += C[j]
#%% Visualization of RISH features up to 6 order      
s_axial = 70

plt.subplot(1,4,1)
plt.imshow(np.rot90(RISH[0][:,:,s_axial],1), cmap='gray')
plt.axis("off") 
plt.title("0 order")
plt.subplot(1,4,2 )
plt.imshow(np.rot90(RISH[1][:,:,s_axial],1), cmap='gray')
plt.axis("off") 
plt.title("2 order")
plt.subplot(1,4,3)
plt.imshow(np.rot90(RISH[2][:,:,s_axial],1), cmap='gray') 
plt.axis("off")
plt.title("4 order") 
plt.subplot(1,4,4)
plt.imshow(np.rot90(RISH[3][:,:,s_axial],1), cmap='gray')   
plt.axis("off") 
plt.title("6 order")


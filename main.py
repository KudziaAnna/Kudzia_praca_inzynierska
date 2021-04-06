import numpy as np
import matplotlib.pyplot as plt
from measures import RTOP_ISBI2019, single_shell_MRI2018

# niftii support
import nibabel as nib

# DIPY library 
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti

def remove_spines(plot_var):
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
        

#%% Data loading - HCP WuMinn
subject = '200109'

data_dir = '/home/tomasz/data/brain_diffusion/hcp_wuminn/'
filename_data =  data_dir + subject + '/T1w/Diffusion/data.nii'
filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'
filename_mask =  data_dir + subject + '/fsl_' + subject + '_data_brain_mask.nii.gz'

# load HCP data
img = nib.load(filename_data)
data_volume = img.get_data()

# load b-values and vectors
bvals_original, bvecs_original = read_bvals_bvecs(filename_bvals, filename_bvecs)

# load binary mask
mask_dict = nib.load(filename_mask)
data_mask = mask_dict.get_data()

# select slice and data
slice = 60

idx0 = np.where(bvals_original < 100)[0]
idx1000 = np.where(np.logical_and(bvals_original < 1100, bvals_original > 900))[0]
idx2000 = np.where(np.logical_and(bvals_original < 2100, bvals_original > 1900))[0]
idx3000 = np.where(np.logical_and(bvals_original < 3100, bvals_original > 2900))[0]

# 
S0 = np.mean(data_volume[:,:,slice,idx0], axis=2)
S1000 = data_volume[:,:,slice,idx1000]
S2000 = data_volume[:,:,slice,idx2000]
S3000 = data_volume[:,:,slice,idx3000]
mask = data_mask[:,:,slice]


# GradientTable object
big_delta = 0.0431       # valid for HCP WU-Minn data only
small_delta = 0.0106     # valid for HCP WU-Minn data only

tau = big_delta - small_delta/3


#%% Data plotting

# DTI at b = 1000
gtab = gradient_table(bvals=bvals_original[np.concatenate((idx0, idx1000))],
                      bvecs=bvecs_original[np.concatenate((idx0, idx1000)),:])   

# fiting procedure
tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(data_volume[:,:,slice, np.concatenate((idx0, idx1000))])  
MD = dti.mean_diffusivity(tenfit.evals)
FA = dti.fractional_anisotropy(tenfit.evals)


# single-shell measures at b = 2000
RTOP = RTOP_ISBI2019(S2000, S0, bval=2000, tau=tau, fwhm=0)
ASD, CVD = single_shell_MRI2018(S2000, S0, bval=2000, fwhm=0)


plt.subplot(231)
plt.imshow(np.rot90(MD*mask, 1), vmin=0, vmax=3e-3, cmap='gray')
plt.title('MD')

plt.subplot(232)
plt.imshow(np.rot90(FA*mask, 1), vmin=0, vmax=1, cmap='gray')
plt.title('FA')

plt.subplot(234)
plt.imshow(np.rot90(RTOP**(1/3)*mask, 1), vmin=30, vmax=80, cmap='hot')
plt.title('$\mathrm{RTOP}^{1/3}$')

plt.subplot(235)
plt.imshow(np.rot90(ASD*mask, 1), vmin=0, vmax=3e-3, cmap='gray')
plt.title('ASD')

plt.subplot(236)
plt.imshow(np.rot90(CVD*mask, 1), vmin=0, vmax=0.5, cmap='gray')
plt.title('CVD')



import matplotlib
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from SHBasis import SHBasis
from SHResNet_cross import SHResNet
from measures import RTOP_ISBI2019, single_shell_MRI2018

# niftii support
import nibabel as nib

# DIPY library
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.reconst.odf import gfa

font = {'size'   : 25}

matplotlib.rc('font', **font)

def predict(args, input_size, model_name ='/home/kudzia/SHResNet/models/SHResNet_block_1_2_10.pt', normalized=False):
    data_dir = '/home/kudzia/data/'
    subject = str(args.subject)
    filename_data = data_dir + subject + '/T1w/Diffusion/data.nii'
    filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
    filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'

    true_x = SHBasis(filename_data, filename_bvals, filename_bvecs, 4, 1000, normalized=False)
    data_x = true_x.get_SHCoeff()
    for i in range(1, data_x.shape[1]-1):
      for j in range(1, data_x.shape[2]-1):
        for k in range(1, data_x.shape[3]-1):
          if i == 1:
            data = data_x[:,:3, :3, :3].reshape((1, data_x.shape[0], 3, 3, 3))
          else:
            tmp_i = data_x[:, i-1:i+2, j-1:j+2, k-1:k+2].reshape((1, data_x.shape[0], 3, 3, 3))
            data = np.append(data, tmp_i, axis=0)
    
    print(data.shape)
    model = SHResNet(input_size, 2)
    model.double().cuda()
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['mode_state_dict'])
    model.eval()     
    with torch.no_grad():
      data = torch.from_numpy(data).cuda()
      prediction = model(data)
    
    harm_SHCoeff = prediction.data.cpu()[1:]
    print(harm_SHCoeff.shape)
    harm_SHCoeff = np.transpose(harm_SHCoeff.reshape((prediction.shape[0]-1, prediction.shape[1])))
    print(harm_SHCoeff.shape)
    harm_SHCoeff = harm_SHCoeff.reshape(data_x.shape[0], data_x.shape[1]-3, data_x.shape[2]-2, -1)
    print(harm_SHCoeff.shape)
    '''
    rish_x = true_x.get_RISH_from_SHCoeff(data_x, args.order )
    rish_harm = true_x.get_RISH_from_SHCoeff(harm_SHCoeff.numpy(), args.order)
    harm_SHCoeff = data_x *np.sqrt(rish_harm)/np.sqrt(rish_x)
    harm_SHCoeff[np.isnan(harm_SHCoeff)] = 0
    harm_SHCoeff[np.abs(harm_SHCoeff) == np.inf] =0
    '''
    harm_x = true_x.get_SHdata_from_SHCoeff(harm_SHCoeff)
    
    return harm_x

def get_figure(args, harm_x):
    data_dir = '/home/kudzia/data/'
    subject = str(args.subject)
    filename_data = data_dir + subject + '/T1w/Diffusion/data.nii'
    filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
    filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'
    
    true_x = SHBasis(filename_data, filename_bvals, filename_bvecs, args.order, args.b_y)
    data = true_x.data
    
    print("Data fig max: "+str(np.max(data)))
    print("Data fig min: "+str(np.min(data)))
    print("Harm fig max: "+str(np.max(harm_x)))
    print("Harm fig min: "+str(np.min(harm_x)))

    fig, axs = plt.subplots(2, 3, figsize=(25, 15))
    im1 = axs[0,0].imshow(np.rot90(data[:, :, 50, 10], 1), cmap='gray')
    axs[0,0].axis("off")
    axs[0,0].title.set_text('Before harmonization')
    fig.colorbar(im1, ax=axs[0,0])
    
    im2 = axs[0,1].imshow(np.rot90(harm_x[:, :, 0, 10], 1), cmap='gray')
    axs[0,1].axis("off")
    axs[0,1].title.set_text('After harmonization')
    fig.colorbar(im2, ax=axs[0,1])
    
    im3 = axs[0,2].imshow(np.rot90(np.abs(harm_x[:, :, 0, 10] - data[:72, :98, 50, 10]), 1), cmap='gray')
    axs[0,2].axis("off")
    axs[0,2].title.set_text('Absolute Error')
    fig.colorbar(im3, ax=axs[0,2])
    
    fig.savefig("/home/kudzia/results/SHResNet/data_cross_" + args.net + "_" + str(args.b_x) + "_" + str(args.b_y) + ".svg")
    
def get_norm_figure(args, harm_x):
    data_dir = '/home/kudzia/data/'
    subject = str(args.subject)
    filename_data = data_dir + subject + '/T1w/Diffusion/data.nii'
    filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
    filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'

    bvals, bvecs = read_bvals_bvecs(filename_bvals, filename_bvecs)

    true_x = SHBasis(filename_data, filename_bvals, filename_bvecs, args.order, args.b_y, normalized=True)
    data = true_x.data
    data_all = true_x.all_data
    idx0 = np.where(bvals < 100)[0]
    
    data_b0 = np.mean(data_all[:, :, :, idx0], axis=3)
    data_b0 = data_b0[:, :, :, np.newaxis]

    print("Data max: "+str(np.max(data)))
    print("Data min: "+str(np.min(data)))
    print("Harm max: "+str(np.max(harm_x)))
    print("Harm min: "+str(np.min(harm_x)))
    
    fig = plt.figure(figsize=(25, 15))
    ax1 = fig.add_subplot(4, 3, 1)
    ax1.imshow(np.rot90(data[:, :, 60, 10], 1), cmap='gray')
    ax1.axis("off")
    ax1.title.set_text('Normalized before harm')
    
    ax2 = fig.add_subplot(4, 3, 2)
    ax2.imshow(np.rot90(harm_x[:, :, 60, 10], 1), cmap='gray')
    
    ax2.axis("off")
    ax2.title.set_text('Normalized after harm')
    
    ax3 = fig.add_subplot(4, 3, 3)
    ax3.imshow(np.rot90(np.abs(harm_x[:, :, 60, 10] - data[:, :, 60, 10]), 1), cmap='gray')
    ax3.axis("off")
    ax3.title.set_text('Absolute Error')
    
    ax4 = fig.add_subplot(4, 3, 4)
    ax4.imshow(np.rot90(data[:, 60, :, 10], 1), cmap='gray')
    ax4.axis("off")
    
    ax5 = fig.add_subplot(4, 3, 5)
    ax5.imshow(np.rot90(harm_x[:, 60, :, 10], 1), cmap='gray')
    ax5.axis("off")
    
    ax6 = fig.add_subplot(4, 3, 6)
    ax6.imshow(np.rot90(np.abs(harm_x[:, 60, :, 10] - data[:, 60, :, 10]), 1), cmap='gray')
    ax6.axis("off")
    
    ax7 = fig.add_subplot(4, 3, 7)
    ax7.imshow(np.rot90(data[60, :, :, 10], 1), cmap='gray')
    ax7.axis("off")

    ax8 = fig.add_subplot(4, 3, 8)
    ax8.imshow(np.rot90(harm_x[60, :, :, 10], 1), cmap='gray')
    ax8.axis("off")
    
    ax9 = fig.add_subplot(4, 3, 9)
    ax9.imshow(np.rot90(np.abs(harm_x[60, :, :, 10] - data[60, :, :, 10]), 1), cmap='gray')
    ax9.axis("off")
    
    ax10 = fig.add_subplot(4, 3, 10)
    ax10.imshow(np.rot90(data[:, :, 40, 10], 1), cmap='gray')
    ax10.axis("off")
    
    ax11 = fig.add_subplot(4, 3, 11)
    ax11.imshow(np.rot90(harm_x[:, :, 40, 10], 1), cmap='gray')
    ax11.axis("off")
    
    ax12 = fig.add_subplot(4, 3, 12)
    ax12.imshow(np.rot90(np.abs(harm_x[:, :, 40, 10] - data[:, :, 40, 10]), 1), cmap='gray')
    ax12.axis("off")
    
    fig.savefig("/home/kudzia/results/SHResNet/norm_data_cros_" + args.net + "_" + str(args.b_x) + "_" + str(args.b_y) + ".svg")

def FA_MD_true(args):
    data_dir = '/home/kudzia/data/'
    subject = str(args.subject)
    filename_data = data_dir + subject + '/T1w/Diffusion/data.nii'
    filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
    filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'
    filename_mask = data_dir + subject + '/fsl_' + subject + '_data_brain_mask.mat'

    # load HCP data
    img = nib.load(filename_data)
    data_volume = img.get_data()

    # load binary mask
    mask_dict = loadmat(filename_mask)
    data_mask = mask_dict['mask']

    # load b-values and vectors
    bvals_original, bvecs_original = read_bvals_bvecs(filename_bvals, filename_bvecs)

    idx0 = np.where(bvals_original < 100)[0]
    idx1000 = np.where(np.logical_and(bvals_original < 1100, bvals_original > 900))[0]
    mask = data_mask[35:110, 35:135, args.slice]

    # DTI at b = 1000
    gtab = gradient_table(bvals=bvals_original[np.concatenate((idx0, idx1000))],
                            bvecs=bvecs_original[np.concatenate((idx0, idx1000)), :])

    # fiting procedure
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data_volume[35:110, 35:135, args.slice, np.concatenate((idx0, idx1000))])
    MD = dti.mean_diffusivity(tenfit.evals)
    FA = dti.fractional_anisotropy(tenfit.evals)

    return FA*mask, MD*mask


def FA_MD_pred(args, prediction):
    data_dir = '/home/kudzia/data/'
    subject = str(args.subject)
    filename_data = data_dir + subject + '/T1w/Diffusion/data.nii'
    filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
    filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'
    filename_mask = data_dir + subject + '/fsl_' + subject + '_data_brain_mask.mat'

    if args.b_x != 1000:
        return 'cannot calculate MD and FA for b different from 1000 '
    else:
        # load HCP data
        img = nib.load(filename_data)
        data = img.get_fdata()
        data_volume = prediction[:, :, args.slice, :]

        # load binary mask
        mask_dict = loadmat(filename_mask)
        data_mask = mask_dict['mask']

        # load b-values and vectors
        bvals_original, bvecs_original = read_bvals_bvecs(filename_bvals, filename_bvecs)

        idx0 = np.where(bvals_original < 100)[0]
        idx1000 = np.where(bvals_original == 1000)[0]
        mask = data_mask[35:110, 35:135, args.slice]
        data = data[35:110, 35:135, args.slice, idx0]
        data_volume = np.concatenate((data, data_volume), axis=2)

        # DTI at b = 1000
        gtab = gradient_table(bvals=bvals_original[np.concatenate((idx0, idx1000))],
                              bvecs=bvecs_original[np.concatenate((idx0, idx1000)), :])

        # fiting procedure
        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(data_volume[:, :, :])
        MD = dti.mean_diffusivity(tenfit.evals)
        FA = dti.fractional_anisotropy(tenfit.evals)
        
        print(FA*mask)

        return FA*mask, MD*mask


def single_shelL_measurments_true(args, bvalue):
    # %% Data loading - HCP WuMinn
    subject = str(args.subject)
    data_dir = '/home/kudzia/data/'
    filename_data = data_dir + subject + '/T1w/Diffusion/data.nii'
    filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
    filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'
    

    # load HCP data
    original = SHBasis(filename_data, filename_bvals, filename_bvecs, args.order, bvalue)
    data_volume = original.all_data
    mask = original.mask[:, : , args.slice]
    
    # load b-values and vectors
    bvals_original, bvecs_original = read_bvals_bvecs(filename_bvals, filename_bvecs)

    idx0 = np.where(bvals_original < 100)[0]
    S0 = np.mean(data_volume[:, :, args.slice, idx0], axis=2)
    
    if bvalue == 1000:
        idx1000 = np.where(bvals_original == 1000)[0]
        data = data_volume[:, :, args.slice, idx1000]
    elif bvalue == 2000:
        idx2000 = np.where(bvals_original == 2000)[0]
        data = data_volume[:, :, args.slice, idx2000]
    else:
        idx3000 = np.where(bvals_original == 3000)[0]
        data = data_volume[:, :, args.slice, idx3000]

    
    # GradientTable object
    big_delta = 0.0431  # valid for HCP WU-Minn data only
    small_delta = 0.0106  # valid for HCP WU-Minn data only

    tau = big_delta - small_delta / 3

    # single-shell measures 
    RTOP, GFA = RTOP_ISBI2019(data, S0, args.b_y, tau=tau, fwhm=0)
    ASD, CVD = single_shell_MRI2018(data, S0, args.b_y, fwhm=0)

    return (RTOP**(1/3)*mask, ASD*mask, CVD*mask, GFA*mask)


def single_shelL_measurments_prediction(args, prediction):
    # %% Data loading - HCP WuMinn
    subject = str(args.subject)
    data_dir = '/home/kudzia/data/'
    filename_data = data_dir + subject + '/T1w/Diffusion/data.nii'
    filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
    filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'

    # load HCP data
    data_volume = prediction

    original = SHBasis(filename_data, filename_bvals, filename_bvecs, args.order, 0)
    data= original.all_data
    mask = original.mask[:, : , args.slice]
    
    # load b-values and vectors
    bvals_original, bvecs_original = read_bvals_bvecs(filename_bvals, filename_bvecs)

    idx0 = np.where(bvals_original < 100)[0]
    S0 = np.mean(data[:, :, args.slice, idx0], axis=2)

    data = data_volume[:, :, args.slice, :]

    # GradientTable object
    big_delta = 0.0431  # valid for HCP WU-Minn data only
    small_delta = 0.0106  # valid for HCP WU-Minn data only

    tau = big_delta - small_delta / 3

    # single-shell measures
    RTOP, GFA = RTOP_ISBI2019(data, S0, args.b_x, tau=tau, fwhm=0, pred=True)
    ASD, CVD = single_shell_MRI2018(data, S0, args.b_x, fwhm=0, pred=True)
    
    return (RTOP**(1/3)*mask, ASD*mask, CVD*mask, GFA*mask)

def visual_measures(args, prediction, figure=True):
    RTOP_true_x, ASD_true_x, CVD_true_x, GFA_true_x = single_shelL_measurments_true(args, args.b_x)
    RTOP_true_y, ASD_true_y, CVD_true_y, GFA_true_y = single_shelL_measurments_true(args, args.b_y)
    RTOP_harm, ASD_harm, CVD_harm, GFA_harm = single_shelL_measurments_prediction(args, prediction)
    RTOP_err_before = np.abs(RTOP_true_x - RTOP_true_y)
    RTOP_sqerr_before = np.power((RTOP_true_x - RTOP_true_y), 2)
    RTOP_err_after = np.abs(RTOP_true_x - RTOP_harm)
    RTOP_sqerr_after = np.power((RTOP_true_x - RTOP_harm), 2)
    RTOP_mre_before =  RTOP_err_before/RTOP_true_y
    RTOP_mre_after =  RTOP_err_after/RTOP_harm
    RTOP_mre_before[RTOP_mre_before == np.inf] = 0
    RTOP_mre_after[RTOP_mre_after == np.inf] = 0
    RTOP_mre_before[np.isnan(RTOP_mre_before)] = 0
    RTOP_mre_after[np.isnan(RTOP_mre_after)] = 0

    ASD_err_before = np.abs(ASD_true_x - ASD_true_y)
    ASD_sqerr_before = np.power((ASD_true_x - ASD_true_y), 2)
    ASD_err_after = np.abs(ASD_true_x - ASD_harm)
    ASD_sqerr_after = np.power((ASD_true_x - ASD_harm), 2)
    ASD_mre_before =  ASD_err_before/ASD_true_y
    ASD_mre_after =  ASD_err_after/ASD_harm
    ASD_mre_before[ASD_mre_before == np.inf] = 0
    ASD_mre_after[ASD_mre_after == np.inf] = 0
    ASD_mre_before[np.isnan(ASD_mre_before)] = 0
    ASD_mre_after[np.isnan(ASD_mre_after)] = 0

    CVD_err_before = np.abs(CVD_true_x - CVD_true_y)
    CVD_sqerr_before = np.power((CVD_true_x - CVD_true_y), 2)
    CVD_err_after = np.abs(CVD_true_x - CVD_harm)
    CVD_sqerr_after = np.power((CVD_true_x - CVD_harm), 2)
    CVD_mre_before =  CVD_err_before/CVD_true_y
    CVD_mre_after =  CVD_err_after/CVD_harm
    CVD_mre_before[CVD_mre_before == np.inf] = 0
    CVD_mre_after[CVD_mre_after == np.inf] = 0
    CVD_mre_before[np.isnan(CVD_mre_before)] = 0
    CVD_mre_after[np.isnan(CVD_mre_after)] = 0

    GFA_err_before = np.abs(GFA_true_x - GFA_true_y)
    GFA_sqerr_before = np.power((GFA_true_x - GFA_true_y), 2)
    GFA_err_after = np.abs(GFA_true_x - GFA_harm)
    GFA_sqerr_after = np.power((GFA_true_x - GFA_harm), 2)
    GFA_mre_before =  GFA_err_before/GFA_true_y
    GFA_mre_after =  GFA_err_after/GFA_harm
    GFA_mre_before[GFA_mre_before == np.inf] = 0
    GFA_mre_after[GFA_mre_after == np.inf] = 0
    GFA_mre_before[np.isnan(GFA_mre_before)] = 0
    GFA_mre_after[np.isnan(GFA_mre_after)] = 0

    if figure:
        fig3 = plt.figure(figsize=(15, 15))
        ax1 = fig3.add_subplot(3, 3, 1)
        ax1.imshow(np.rot90(RTOP_true_x, 1), vmin=30, vmax=80, cmap='hot')
        ax1.axis("off")
        ax1.title.set_text('$\mathrm{RTOP}^{1/3}$ b=' + str(args.b_x))
        ax2 = fig3.add_subplot(3, 3, 2)
        ax2.imshow(np.rot90(RTOP_true_y, 1), vmin=30, vmax=80, cmap='hot')
        ax2.axis("off")
        ax2.title.set_text('$\mathrm{RTOP}^{1/3}$ b=' + str(args.b_y))
        ax3 = fig3.add_subplot(3, 3, 3)
        ax3.imshow(np.rot90(RTOP_harm, 1), vmin=30, vmax=80, cmap='hot')
        ax3.axis("off")
        ax3.title.set_text('$\mathrm{RTOP}^{1/3}$ harmonized')
        ax4 = fig3.add_subplot(3, 3, 4)
        ax4.imshow(np.rot90(RTOP_err_before, 1), vmin=30, vmax=80, cmap='hot')
        ax4.axis("off")
        ax4.title.set_text('$|\mathrm{RTOP}^{1/3}|$ before')
        ax5 = fig3.add_subplot(3, 3, 5)
        ax5.imshow(np.rot90(RTOP_sqerr_before, 1), vmin=30, vmax=80, cmap='hot')
        ax5.axis("off")
        ax5.title.set_text('$|\mathrm{RTOP}^{1/3}|^2$ before')
        ax6 = fig3.add_subplot(3, 3, 7)
        ax6.imshow(np.rot90(RTOP_err_after, 1), vmin=30, vmax=80, cmap='hot')
        ax6.axis("off")
        ax6.title.set_text('$|\mathrm{RTOP}^{1/3}|$ after')
        ax7 = fig3.add_subplot(3, 3, 8)
        ax7.imshow(np.rot90(RTOP_sqerr_after, 1), vmin=30, vmax=80, cmap='hot')
        ax7.axis("off")
        ax7.title.set_text('$|\mathrm{RTOP}^{1/3}|^2$ after')
        fig3.savefig("/home/kudzia/results/SHResNet/result_RTOP_cross_red_" + args.net + "_" + str(args.b_x) + "_" + str(args.b_y) + ".svg")

        fig4 = plt.figure(figsize=(15, 15))
        ax1 = fig4.add_subplot(3, 3, 1)
        ax1.imshow(np.rot90(ASD_true_x, 1), vmin=0, vmax=3e-3, cmap='gray')
        ax1.axis("off")
        ax1.title.set_text('ASD b=' + str(args.b_x))
        ax2 = fig4.add_subplot(3, 3, 2)
        ax2.imshow(np.rot90(ASD_true_y, 1), vmin=0, vmax=3e-3, cmap='gray')
        ax2.axis("off")
        ax2.title.set_text('ASD b=' + str(args.b_y))
        ax3 = fig4.add_subplot(3, 3, 3)
        ax3.imshow(np.rot90(ASD_harm, 1), vmin=0, vmax=3e-3, cmap='gray')
        ax3.axis("off")
        ax3.title.set_text('ASD harmonized')
        ax4 = fig4.add_subplot(3, 3, 4)
        ax4.imshow(np.rot90(ASD_err_before, 1), vmin=0, vmax=3e-3, cmap='gray')
        ax4.axis("off")
        ax4.title.set_text('|ASD| before')
        ax5 = fig4.add_subplot(3, 3, 5)
        ax5.imshow(np.rot90(ASD_sqerr_before, 1), vmin=0, vmax=3e-3, cmap='gray')
        ax5.axis("off")
        ax5.title.set_text('$|\mathrm{ASD}|^2$ before')
        ax6 = fig4.add_subplot(3, 3, 7)
        ax6.imshow(np.rot90(ASD_err_after, 1), vmin=0, vmax=3e-3, cmap='gray')
        ax6.axis("off")
        ax6.title.set_text('|ASD| after')
        ax7 = fig4.add_subplot(3, 3, 8)
        ax7.imshow(np.rot90(ASD_sqerr_after, 1), vmin=0, vmax=3e-3, cmap='gray')
        ax7.axis("off")
        ax7.title.set_text('$|\mathrm{ASD}|^2$ after')
        fig4.savefig("/home/kudzia/results/SHResNet/result_ASD_cross_red_" + args.net + "_" + str(args.b_x) + "_" + str(args.b_y) + ".svg")

        fig5 = plt.figure(figsize=(15, 15))
        ax1 = fig5.add_subplot(3, 3, 1)
        ax1.imshow(np.rot90(CVD_true_x, 1), vmin=0, vmax=0.5, cmap='gray')
        ax1.axis("off")
        ax1.title.set_text('CVD b=' + str(args.b_x))
        ax2 = fig5.add_subplot(3, 3, 2)
        ax2.imshow(np.rot90(CVD_true_y, 1), vmin=0, vmax=0.5, cmap='gray')
        ax2.axis("off")
        ax2.title.set_text('CVD b=' + str(args.b_y))
        ax3 = fig5.add_subplot(3, 3, 3)
        ax3.imshow(np.rot90(CVD_harm, 1), vmin=0, vmax=0.5, cmap='gray')
        ax3.axis("off")
        ax3.title.set_text('CVD harmonized')
        ax4 = fig5.add_subplot(3, 3, 4)
        ax4.imshow(np.rot90(CVD_err_before, 1), vmin=0, vmax=0.5, cmap='gray')
        ax4.axis("off")
        ax4.title.set_text('|CVD| before')
        ax5 = fig5.add_subplot(3, 3, 5)
        ax5.imshow(np.rot90(CVD_sqerr_before, 1), vmin=0, vmax=0.5, cmap='gray')
        ax5.axis("off")
        ax5.title.set_text('$|\mathrm{CVD}|^2$ before')
        ax6 = fig5.add_subplot(3, 3, 7)
        ax6.imshow(np.rot90(CVD_err_after, 1), vmin=0, vmax=0.5, cmap='gray')
        ax6.axis("off")
        ax6.title.set_text('|CVD| after')
        ax7 = fig5.add_subplot(3, 3, 8)
        ax7.imshow(np.rot90(CVD_sqerr_after, 1), vmin=0, vmax=0.5, cmap='gray')
        ax7.axis("off")
        ax7.title.set_text('$|\mathrm{CVD}|^2$ after')
        fig5.savefig("/home/kudzia/results/SHResNet/result_CVD_cross_red_" + args.net + "_" + str(args.b_x) + "_" + str(args.b_y) + ".svg")

        fig6 = plt.figure(figsize=(15, 15))
        ax1 = fig6.add_subplot(3, 3, 1)
        ax1.imshow(np.rot90(GFA_true_x, 1), vmin=0, vmax=0.5, cmap='gray')
        ax1.axis("off")
        ax1.title.set_text('GFA b=' + str(args.b_x))
        ax2 = fig6.add_subplot(3, 3, 2)
        ax2.imshow(np.rot90(GFA_true_y, 1), vmin=0, vmax=0.5, cmap='gray')
        ax2.axis("off")
        ax2.title.set_text('GFA b=' + str(args.b_y))
        ax3 = fig6.add_subplot(3, 3, 3)
        ax3.imshow(np.rot90(GFA_harm, 1), vmin=0, vmax=0.5, cmap='gray')
        ax3.axis("off")
        ax3.title.set_text('GFA harmonized')
        ax4 = fig6.add_subplot(3, 3, 4)
        ax4.imshow(np.rot90(GFA_err_before, 1), vmin=0, vmax=0.5, cmap='gray')
        ax4.axis("off")
        ax4.title.set_text('|GFA| before')
        ax5 = fig6.add_subplot(3, 3, 5)
        ax5.imshow(np.rot90(GFA_sqerr_before, 1), vmin=0, vmax=0.5, cmap='gray')
        ax5.axis("off")
        ax5.title.set_text('$|\mathrm{GFA}|^2$ before')
        ax6 = fig6.add_subplot(3, 3, 7)
        ax6.imshow(np.rot90(GFA_err_after, 1), vmin=0, vmax=0.5, cmap='gray')
        ax6.axis("off")
        ax6.title.set_text('|GFA| after')
        ax7 = fig6.add_subplot(3, 3, 8)
        ax7.imshow(np.rot90(GFA_sqerr_after, 1), vmin=0, vmax=0.5, cmap='gray')
        ax7.axis("off")
        ax7.title.set_text('$|\mathrm{GFA}|^2$ after')
        fig6.savefig("/home/kudzia/results/SHResNet/result_GFA_cross_red_" + args.net + "_" + str(args.b_x) + "_" + str(args.b_y) + ".svg")
      

    if args.b_y == 1000:
        FA_x, MD_x = FA_MD_true(args)
        FA_harm, MD_harm = FA_MD_pred(args, prediction)
        FA_err = np.abs(FA_x - FA_harm)
        FA_sqerr = np.power((FA_x - FA_harm), 2)
        MD_err = np.abs(MD_x - MD_harm)
        MD_sqerr = np.power((MD_x - MD_harm), 2)
        if figure:
          fig1 = plt.figure(figsize=(15, 15))
          ax1 = fig1.add_subplot(2, 2, 1)
          ax1.imshow(np.rot90(FA_x, 1), vmin=0, vmax=1, cmap='gray')
          ax1.axis("off")
          ax1.title.set_text("FA original")
          ax2 = fig1.add_subplot(2, 2, 2)
          ax2.imshow(np.rot90(FA_harm, 1), vmin=0, vmax=1, cmap='gray')
          ax2.axis("off")
          ax2.title.set_text("FA harmonized")
          ax3 = fig1.add_subplot(2, 2, 3)
          ax3.imshow(np.rot90(FA_err, 1), vmin=0, vmax=1, cmap='gray')
          ax3.axis("off")
          ax3.title.set_text("|FA|")
          ax4 = fig1.add_subplot(2, 2, 4)
          ax4.imshow(np.rot90(FA_sqerr, 1), vmin=0, vmax=1, cmap='gray')
          ax4.axis("off")
          ax4.title.set_text(r'$ |\mathrm{FA}|^{2}$')
          fig1.savefig("/home/kudzia/results/SHResNet/result_FA_" + args.net + "_" + str(args.b_x) + "_" + str(args.b_y) + ".svg")

          fig2 = plt.figure(figsize=(15, 15))
          ax1 = fig2.add_subplot(2, 2, 1)
          ax1.imshow(np.rot90(MD_x, 1), vmin=0, vmax=3e-3, cmap='gray')
          ax1.axis("off")
          ax1.title.set_text("MD original")
          ax2 = fig2.add_subplot(2, 2, 2)
          ax2.imshow(np.rot90(MD_harm, 1), vmin=0, vmax=3e-3, cmap='gray')
          ax2.axis("off")
          ax2.title.set_text("MD harmonized")
          ax3 = fig2.add_subplot(2, 2, 3)
          ax3.imshow(np.rot90(MD_err, 1), vmin=0, vmax=3e-3, cmap='gray')
          ax3.axis("off")
          ax3.title.set_text("|MD|")
          ax4 = fig2.add_subplot(2, 2, 4)
          ax4.imshow(np.rot90(MD_sqerr, 1), vmin=0, vmax=3e-3, cmap='gray')
          ax4.axis("off")
          ax4.title.set_text(r'$ |\mathrm{MD}|^{2}$')
          fig2.savefig("/home/kudzia/results/SHResNet/result_MD_" + args.net + "_" + str(args.b_x) + "_" + str(args.b_y) + ".svg")

        return FA_err, MD_sqerr, RTOP_sqerr_before, RTOP_sqerr_after, ASD_sqerr_before, ASD_sqerr_after, \
                      CVD_sqerr_before, CVD_serr_after, GFA_sqerr_before, GFA_sqerr_after
    else:
        return RTOP_mre_before, RTOP_mre_after, ASD_mre_before, ASD_mre_after, \
                      CVD_mre_before, CVD_mre_after, GFA_mre_before, GFA_mre_after

def epoch_measures(args, input_size):
    model_dir = '/home/kudzia/'+str(args.net)+'/models/'
    if args.b_y == 1000:
        FA = []
        MD = []
        RTOP_b = []
        ASD_b = []
        CVD_b = []
        GFA_b = []
        RTOP_a = []
        ASD_a = []
        CVD_a = []
        GFA_a = []
        
        for i in range(args.num_epochs):
            model = model_dir+str(args.net)+'_'+str(int(args.b_x/1000))+'_'+str(int(args.b_y/1000))+'_'+str((i+1)*5)+'.pt'
            prediction = predict(args, input_size, model)
            errors = visual_measures(args, prediction, figure=False)

            FA.append(np.mean(errors[0]))
            MD.append(np.mean(errors[1]))
            RTOP_b.append(np.mean(errors[2]))
            RTOP_a.append(np.mean(errors[3]))
            ASD_b.append(np.mean(errors[4]))
            ASD_a.append(np.mean(errors[5]))
            CVD_b.append(np.mean(errors[6]))
            CVD_a.append(np.mean(errors[7]))
            GFA_b.append(np.mean(errors[8]))
            GFA_a.append(np.mean(errors[9]))
        

        # set width of bar
        barWidth = 1.0
        fig = plt.subplots(figsize=(20, 15))

        # Set position of bar on X axis
        br1 = np.arange(args.num_epochs)*16
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        
        br4 = [x + barWidth for x in br3]
        br5 = [x + barWidth for x in br4]
        br6 = [x + barWidth for x in br5]
        
        br7 = [x + barWidth for x in br6]
        br8 = [x + barWidth for x in br7]
        br9 = [x + barWidth for x in br8]
        
        br10 = [x + barWidth for x in br9]
        br11 = [x + barWidth for x in br10]
        br12 = [x + barWidth for x in br11]
        
        br13 = [x + barWidth for x in br12]
        br14 = [x + barWidth for x in br13]

        # Make the plot
        plt.bar(br1, FA, color='r', width=barWidth,
                edgecolor='grey', label='FA')
        plt.bar(br2, MD, color='g', width=barWidth,
                edgecolor='grey', label='MD')
        plt.bar(br4, RTOP_b, color='b', width=barWidth,
                edgecolor='grey', label='RTOP before')
        plt.bar(br5, RTOP_a, color='aqua', width=barWidth,
                edgecolor='grey', label='RTOP after')
        plt.bar(br7, ASD_b, color='y', width=barWidth,
                edgecolor='grey', label='ASD before')
        plt.bar(br8, ASD_a, color='orange', width=barWidth,
                edgecolor='grey', label='ASD after')    
        plt.bar(br11, CVD_b, color='chocolate', width=barWidth,
                edgecolor='grey', label='CVD before')
        plt.bar(br12, CVD_a, color='sienna', width=barWidth,
                edgecolor='grey', label='CVD after')
                        
        plt.bar(br13, GFA_b, color='deeppink', width=barWidth,
                edgecolor='gray', label='GFA before')
        plt.bar(br14, GFA_a, color='purple', width=barWidth,
                edgecolor='gray', label='GFA after')
        
    else:
        RTOP_b = []
        ASD_b = []
        CVD_b = []
        GFA_b = []
        RTOP_a = []
        ASD_a = []
        CVD_a = []
        GFA_a = []
        for i in range(args.num_epochs):
            model = model_dir + str(args.net) + '_block_' + str(int(args.b_x/1000)) + '_' + str(int(args.b_y/1000)) + '_' + str(
                (i + 1) * 10) + '.pt'
            prediction = predict(args, input_size, model, normalized=True)
            errors = visual_measures(args, prediction, figure=False)
            RTOP_b.append(np.mean(errors[0]))
            RTOP_a.append(np.mean(errors[1]))
            ASD_b.append(np.mean(errors[2]))
            ASD_a.append(np.mean(errors[3]))
            CVD_b.append(np.mean(errors[4]))
            CVD_a.append(np.mean(errors[5]))
            GFA_b.append(np.mean(errors[6]))
            GFA_a.append(np.mean(errors[7]))
        

        # set width of bar
        barWidth = 2.0
        fig = plt.subplots(figsize=(30, 15))

        # Set position of bar on X axis
        br1 = np.arange(args.num_epochs)*26
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        
        br4 = [x + barWidth for x in br3]
        br5 = [x + barWidth for x in br4]
        br6 = [x + barWidth for x in br5]
        
        br7 = [x + barWidth for x in br6]
        br8 = [x + barWidth for x in br7]
        br9 = [x + barWidth for x in br8]
        
        br10 = [x + barWidth for x in br9]
        br11 = [x + barWidth for x in br10]


        # Make the plot

        plt.bar(br1, RTOP_b, color='b', width=barWidth,
                edgecolor='grey', label='$|\mathrm{RTOP}^{1/3}|^2$ before')
        plt.bar(br2, RTOP_a, color='aqua', width=barWidth,
                edgecolor='grey', label='$|\mathrm{RTOP}^{1/3}|^2$ after')
        plt.bar(br4, ASD_b, color='y', width=barWidth,
                edgecolor='grey', label='ASD before')
        plt.bar(br5, ASD_a, color='orange', width=barWidth,
                edgecolor='grey', label='ASD after')      
   
        plt.bar(br7, CVD_b, color='chocolate', width=barWidth,
                edgecolor='grey', label='CVD before')
        plt.bar(br8, CVD_a, color='sienna', width=barWidth,
                edgecolor='grey', label='CVD after')     
        plt.bar(br10, GFA_b, color='deeppink', width=barWidth,
                edgecolor='gray', label='GFA before')
        plt.bar(br11, GFA_a, color='purple', width=barWidth,
                edgecolor='gray', label='GFA after')

    # Adding Xticks
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('MRE', fontweight='bold')
    plt.xticks([r*26 + 5*barWidth for r in range(len(RTOP_a))],
          ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
          
    # , '55', '60', '65', '70', '75', '80', '85', '90', '95', '100'
    
    plt.savefig("/home/kudzia/results/SHResNet/barplot_per_epoch"+str(args.net)+'_'+str(args.b_x)+'_'+str(args.b_y)+".svg")


def main():
    # Experiment settings
    parser = argparse.ArgumentParser(description='Harmoniation experiments')
    parser.add_argument('-pe', '--per_epoch', type =bool, default=False, help='show barplot per epoch')
    parser.add_argument('-ne', '--num_epochs', type =int, default=10, help='')
    parser.add_argument('-o', '--order', type=int, default=4, help='SH order (default: 4)')
    parser.add_argument('-bx', '--b_x', type=int, default=1000, help='b_x(default: 1000)')
    parser.add_argument('-by', '--b_y', type=int, default=2000, help='b_y(default: 2000)')
    parser.add_argument('-s', '--subject', default=122317)
    parser.add_argument('-sl', '--slice', type=int, default=50, help='slice to show')

    args = parser.parse_args()


    if args.order == 2:
        input_size = 5
    elif args.order == 4:
        input_size = 15
    else:
        input_size = 28
    '''  
    if args.per_epoch:
      epoch_measures(args, input_size=input_size)
    else:
      prediction = predict(args, input_size, normalized=True)
      visual_measures(args, prediction)
      
    '''
    prediction = predict(args, input_size)
    get_figure(args, prediction)
    # prediction = predict(args, input_size, normalized=True)
    # get_norm_figure(args, prediction)

    
if __name__ == '__main__':
    main()
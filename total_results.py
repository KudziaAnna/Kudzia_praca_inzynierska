import matplotlib
import scipy
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from SHBasis import SHBasis
from SHNet import SHNet
# from SHResNet import SHResNet
from measures import RTOP_ISBI2019, single_shell_MRI2018

# niftii support
import nibabel as nib

# DIPY library
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti

font = {'size'   : 25}

matplotlib.rc('font', **font)

def predict(args, input_size, device):
    # data_dir = '/home/tomasz/data/brain_diffusion/hcp_wuminn/'
    data_dir = '/home/kudzia/data/'
    subject = str(args.subject)
    filename_data = data_dir + subject + '/T1w/Diffusion/data.nii'
    filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
    filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'

    true_x = SHBasis(filename_data, filename_bvals, filename_bvecs, args.order, args.b_x)
    true_y = SHBasis(filename_data, filename_bvals, filename_bvecs, args.order, args.b_x)

    data_x = true_x.get_SHCoeff()
    data_y = true_y.get_SHCoeff()

    max_y = np.max(data_y)
    max_x = np.max(data_x)
    data_x = data_x / max_x
    data_x = np.transpose(data_x.reshape((data_x.shape[0], -1)))
    data_x = torch.from_numpy(data_x).to(device)

    if args.net == "SHNet":
        model = SHNet(input_size).to(device)
    else:
        model = SHResNet(input_size, args.num_resblock).to(device)
    model.double()
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['mode_state_dict'])
    model.eval()

    with torch.no_grad():
        harm_SHCoeff = model(data_x)

    harm_SHCoeff = max_y * harm_SHCoeff.reshape((123, 160, 112, 15)).cpu()
    harm_SHCoeff = np.transpose(harm_SHCoeff, (3, 0, 1, 2))

    harm_x = true_x.get_SHdata_from_SHCoeff(harm_SHCoeff)

    return harm_x


def FA_MD_true(args):
    # data_dir = '/home/tomasz/data/brain_diffusion/hcp_wuminn/'
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
    mask = data_mask[10:133, 5:165, args.slice]

    # DTI at b = 1000
    gtab = gradient_table(bvals=bvals_original[np.concatenate((idx0, idx1000))],
                            bvecs=bvecs_original[np.concatenate((idx0, idx1000)), :])

    # fiting procedure
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data_volume[10:133, 5:165, args.slice, np.concatenate((idx0, idx1000))])
    MD = dti.mean_diffusivity(tenfit.evals)
    FA = dti.fractional_anisotropy(tenfit.evals)

    return FA*mask, MD*mask


def FA_MD_pred(args, prediction):
    # data_dir = '/home/tomasz/data/brain_diffusion/hcp_wuminn/'
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
        data = img.get_data()
        data_volume = prediction[:, :, args.slice, :]

        # load binary mask
        mask_dict = loadmat(filename_mask)
        data_mask = mask_dict['mask']

        # load b-values and vectors
        bvals_original, bvecs_original = read_bvals_bvecs(filename_bvals, filename_bvecs)

        idx0 = np.where(bvals_original < 100)[0]
        idx1000 = np.where(bvals_original == 1000)[0]
        mask = data_mask[10:133, 5:165, args.slice]
        data = data[10:133, 5:165, args.slice, idx0]
        data_volume = np.concatenate((data, data_volume), axis=2)

        # DTI at b = 1000
        gtab = gradient_table(bvals=bvals_original[np.concatenate((idx0, idx1000))],
                              bvecs=bvecs_original[np.concatenate((idx0, idx1000)), :])

        # fiting procedure
        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(data_volume[:, :, :])
        MD = dti.mean_diffusivity(tenfit.evals)
        FA = dti.fractional_anisotropy(tenfit.evals)

        return FA*mask, MD*mask


def single_shelL_measurments_true(args, bvalue):
    # %% Data loading - HCP WuMinn
    subject = str(args.subject)
    # data_dir = '/home/tomasz/data/brain_diffusion/hcp_wuminn/'
    data_dir = '/home/kudzia/data/'
    filename_data = data_dir + subject + '/T1w/Diffusion/data.nii'
    filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
    filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'
    filename_mask = data_dir + subject + '/fsl_' + subject + '_data_brain_mask.mat'

    # load HCP data
    img = nib.load(filename_data)
    data_volume = img.get_data()

    # load b-values and vectors
    bvals_original, bvecs_original = read_bvals_bvecs(filename_bvals, filename_bvecs)

    # load binary mask
    mask_dict = loadmat(filename_mask)
    data_mask = mask_dict['mask']

    idx0 = np.where(bvals_original < 100)[0]
    S0 = np.mean(data_volume[10:133, 5:165, args.slice, idx0], axis=2)
    mask = data_mask[10:133, 5:165, args.slice]

    if bvalue == 1000:
        idx1000 = np.where(np.logical_and(bvals_original < 1100, bvals_original > 900))[0]
        data = data_volume[10:133, 5:165, args.slice, idx1000]
    elif bvalue == 2000:
        idx2000 = np.where(np.logical_and(bvals_original < 2100, bvals_original > 1900))[0]
        data = data_volume[10:133, 5:165, args.slice, idx2000]
    else:
        idx3000 = np.where(np.logical_and(bvals_original < 3100, bvals_original > 2900))[0]
        data = data_volume[10:133, 5:165, args.slice, idx3000]

    # GradientTable object
    big_delta = 0.0431  # valid for HCP WU-Minn data only
    small_delta = 0.0106  # valid for HCP WU-Minn data only

    tau = big_delta - small_delta / 3

    # single-shell measures at b = 2000
    RTOP = RTOP_ISBI2019(data, S0, args.b_y, tau=tau, fwhm=0)
    ASD, CVD = single_shell_MRI2018(data, S0, args.b_y, fwhm=0)
    return (RTOP**(1/3)*mask, ASD*mask, CVD*mask)


def single_shelL_measurments_prediction(args, prediction):
    # %% Data loading - HCP WuMinn
    subject = str(args.subject)
    # data_dir = '/home/tomasz/data/brain_diffusion/hcp_wuminn/'
    data_dir = '/home/kudzia/data/'
    filename_data = data_dir + subject + '/T1w/Diffusion/data.nii'
    filename_bvals = data_dir + subject + '/T1w/Diffusion/bvals'
    filename_bvecs = data_dir + subject + '/T1w/Diffusion/bvecs'
    filename_mask = data_dir + subject + '/fsl_' + subject + '_data_brain_mask.mat'

    # load HCP data
    data_volume = prediction
    img = nib.load(filename_data)
    data = img.get_data()

    # load b-values and vectors
    bvals_original, bvecs_original = read_bvals_bvecs(filename_bvals, filename_bvecs)

    # load binary mask
    mask_dict = loadmat(filename_mask)
    data_mask = mask_dict['mask']

    idx0 = np.where(bvals_original < 100)[0]
    S0 = np.mean(data[10:133, 5:165, args.slice, idx0], axis=2)
    mask = data_mask[10:133, 5:165, args.slice]

    data = data_volume[:, :, args.slice, :]

    # GradientTable object
    big_delta = 0.0431  # valid for HCP WU-Minn data only
    small_delta = 0.0106  # valid for HCP WU-Minn data only

    tau = big_delta - small_delta / 3

    # single-shell measures at b = 2000
    RTOP = RTOP_ISBI2019(data, S0, args.b_x, tau=tau, fwhm=0)
    ASD, CVD = single_shell_MRI2018(data, S0, args.b_x, fwhm=0)
    return (RTOP**(1/3)*mask, ASD*mask, CVD*mask)

def visual_measures(args, prediction):
     
    if args.b_x == 1000:
      FA_x, MD_x = FA_MD_true(args)
      FA_harm, MD_harm = FA_MD_pred(args, prediction)
      FA_err = np.abs(FA_x - FA_harm)
      FA_sqerr = np.power((FA_x - FA_harm), 2)

      fig1 = plt.figure(figsize=(15,15))
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
      fig1.savefig("/home/kudzia/results/result_FA_"+args.net+"_"+str(args.b_x)+"_"+str(args.b_y)+".svg")

      MD_err = np.abs(MD_x - MD_harm)
      MD_sqerr = np.power((MD_x - MD_harm), 2)

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
      fig2.savefig("/home/kudzia/results/result_MD_"+args.net+"_"+str(args.b_x)+"_"+str(args.b_y)+".svg")
    

    RTOP_true_x, ASD_true_x, CVD_true_x = single_shelL_measurments_true(args, args.b_x)
    RTOP_true_y, ASD_true_y, CVD_true_y = single_shelL_measurments_true(args, args.b_y)
    RTOP_harm, ASD_harm, CVD_harm = single_shelL_measurments_prediction(args, prediction)
    RTOP_err_before = np.abs(RTOP_true_x - RTOP_true_y)
    RTOP_sqerr_before = np.power((RTOP_true_x - RTOP_true_y), 2)
    RTOP_err_after = np.abs(RTOP_true_x - RTOP_harm)
    RTOP_sqerr_after = np.power((RTOP_true_x - RTOP_harm), 2)

    ASD_err_before = np.abs(ASD_true_x - ASD_true_y)
    ASD_sqerr_before = np.power((ASD_true_x - ASD_true_y), 2)
    ASD_err_after = np.abs(ASD_true_x - ASD_harm)
    ASD_sqerr_after = np.power((ASD_true_x - ASD_harm), 2)

    CVD_err_before = np.abs(CVD_true_x - CVD_true_y)
    CVD_sqerr_before = np.power((CVD_true_x - CVD_true_y), 2)
    CVD_err_after = np.abs(CVD_true_x - CVD_harm)
    CVD_sqerr_after = np.power((CVD_true_x - CVD_harm), 2)

    fig3 = plt.figure(figsize=(15, 15))
    ax1 = fig3.add_subplot(3, 3, 1)
    ax1.imshow(np.rot90(RTOP_true_x, 1),  vmin=30, vmax=80, cmap='hot')
    ax1.axis("off")
    ax1.title.set_text('$\mathrm{RTOP}^{1/3}$ b='+str(args.b_x))
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
    ax7 .imshow(np.rot90(RTOP_sqerr_after, 1), vmin=30, vmax=80, cmap='hot')
    ax7 .axis("off")
    ax7 .title.set_text('$|\mathrm{RTOP}^{1/3}|^2$ after')
    fig3.savefig("/home/kudzia/results/result_RTOP_"+args.net+"_"+str(args.b_x)+"_"+str(args.b_y)+".svg")

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
    fig4.savefig("/home/kudzia/results/result_ASD_"+args.net+"_"+str(args.b_x)+"_"+str(args.b_y)+".svg")

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
    fig5.savefig("/home/kudzia/results/result_CVD_"+args.net+"_"+str(args.b_x)+"_"+str(args.b_y)+".svg")

def main():
    # Experiment settings
    parser = argparse.ArgumentParser(description='Harmoniation experiments')
    parser.add_argument('-n', '--net', default="SHNet", help='net architecture')
    parser.add_argument('-m', '--model', default="model.pt", help='model for predictions')
    parser.add_argument('-o', '--order', type=int, default=4, help='SH order (default: 4)')
    parser.add_argument('-bx', '--b_x', type=int, default=1000, help='b_x(default: 1000)')
    parser.add_argument('-by', '--b_y', type=int, default=3000, help='b_x(default: 3000)')
    parser.add_argument('-rb', '--num_resblock', type=int, default=2, help='slice to show')
    parser.add_argument('-s', '--subject', default=122317)
    parser.add_argument('-sl', '--slice', type=int, default=60, help='slice to show')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.order == 2:
        input_size = 5
    elif args.order == 4:
        input_size = 15
    elif args.order == 6:
        input_size = 28

    prediction = predict(args, input_size, device)
    visual_measures(args, prediction)

if __name__ == '__main__':
    main()
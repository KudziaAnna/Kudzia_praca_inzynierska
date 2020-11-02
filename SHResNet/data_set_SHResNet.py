import torch
import numpy as np
from torch.utils.data import Dataset
from SHBasis import SHBasis

def get_data(b_x, b_y):
    """Initialization"""
    l_order = 4

    data_dir = '/home/kudzia/data/'
    subject = ['111312', '125525', '149741', '172332', '200109', '599671', '660951', '859671', '917255', '122317']
    for i in range(len(subject)):
        print("jestem")
        filename_data = data_dir + subject[i] + '/T1w/Diffusion/data.nii'
        filename_bvals = data_dir + subject[i] + '/T1w/Diffusion/bvals'
        filename_bvecs = data_dir + subject[i] + '/T1w/Diffusion/bvecs'
        sh_first = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order, b_x, normalized=True)
        sh_second = SHBasis(filename_data, filename_bvals, filename_bvecs, l_order, b_y, normalized=True)
        if i == 0:
            sh_coeff_in = sh_first.get_SHCoeff()
            sh_coeff_out = sh_second.get_SHCoeff()
            for i in range(1, sh_coeff_in.shape[1]):
                if i == 1:
                    input = sh_coeff_in[:, 0, :, :].reshape(
                        (1, sh_coeff_in.shape[0], 1, sh_coeff_in.shape[2], sh_coeff_in.shape[3]))
                    label = sh_coeff_out[:, 0, :, :].reshape(
                        (1, sh_coeff_out.shape[0], 1, sh_coeff_out.shape[2], sh_coeff_out.shape[3]))
                else:
                    tmp_i = sh_coeff_in[:, i, :, :].reshape(
                        (1, sh_coeff_in.shape[0], 1, sh_coeff_in.shape[2], sh_coeff_in.shape[3]))
                    tmp_o = sh_coeff_out[:, i, :, :].reshape((1, sh_coeff_out.shape[0], 1,
                                                              sh_coeff_out.shape[2], sh_coeff_out.shape[3]))
                    input = np.append(input, tmp_i, axis=0)
                    label = np.append(label, tmp_o, axis=0)

            X = input
            y = label

        else:
            sh_coeff_in = sh_first.get_SHCoeff()
            sh_coeff_out = sh_second.get_SHCoeff()
            for i in range(1, sh_coeff_in.shape[1]):
                if i == 1:
                    input = sh_coeff_in[:, 0, :, :].reshape(
                        (1, sh_coeff_in.shape[0], 1, sh_coeff_in.shape[2], sh_coeff_in.shape[3]))
                    label = sh_coeff_out[:, 0, :, :].reshape(
                        (1, sh_coeff_out.shape[0], 1, sh_coeff_out.shape[2], sh_coeff_out.shape[3]))
                else:
                    tmp_i = sh_coeff_in[:, i, :, :].reshape(
                        (1, sh_coeff_in.shape[0], 1, sh_coeff_in.shape[2], sh_coeff_in.shape[3]))
                    tmp_o = sh_coeff_out[:, i, :, :].reshape((1, sh_coeff_out.shape[0], 1,
                                                              sh_coeff_out.shape[2], sh_coeff_out.shape[3]))
                    input = np.append(input, tmp_i, axis=0)
                    label = np.append(label, tmp_o, axis=0)

            X = np.concatenate((X, input), axis=0)
            y = np.concatenate((y, label), axis=0)

    if len(X) > len(y):
        X = X[0: len(y)]
    else:
        y = y[0: len(X)]
    
    print("X shape: "+str(X.shape))
    print("y shape: "+str(y.shape))

    return X, y
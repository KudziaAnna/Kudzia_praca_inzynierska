# Author: Tomasz Pieciak (pieciak<at>agh.edu.pl, tpieciak<at>gmail.com)
#         1. Universidad de Valladolid, Valladolid, Spain
#         2. AGH University of Science and Technology, Krakow, Poland
#
import numpy as np
# import multiprocessing as mp
from scipy.ndimage.filters import gaussian_filter
from dipy.reconst.odf import gfa


# -----------------------------------------------------------------------------------------------------------
def RTOP_ISBI2019(data, data_b0, bval, tau, fwhm, pred = False):
    """

    Pieciak T., Bogusz F., Tristán-Vega A., de Luis Garcı́a R., Aja-Fernández S, Single-Shell
    Return-to-the-Origin Probability Diffusion MRI Measure Under a Non-Stationary Rician
    Distributed Noise, IEEE International Symposium on Biomedical Imaging (ISBI), 131-134, 2019

    """

    # prepare data
    if pred:
      Eq = data
    else:
      gradients_no = data.shape[2]
      data_b0 = data_b0[:, :, np.newaxis]
      Eq = np.divide(data, np.repeat(data_b0, gradients_no, axis=2))

    C_tau = (1 / 8) * (np.pi * tau) ** (-3 / 2)

    # correct for numerical instabilities
    Eq[np.isnan(Eq)] = 0
    Eq[Eq > 1] = 1
    Eq[Eq < 1e-3] = 1e-3

    # Eq filtering
    Eq2 = np.copy(Eq)

    if fwhm > 0:
        gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std

        for gg in range(0, gradients_no):
            Eq2[:, :, gg] = gaussian_filter(Eq[:, :, gg], sigma=gauss_std)

    # logarithm the data
    logEq = -np.log(Eq2)
    logEq[np.isnan(logEq)] = 0

    # RTOP calculaton (Pieciak et al., 2019, eq. 7)
    RTOP = (15 / 8) * C_tau * bval ** (3 / 2) * np.mean(logEq ** 2, axis=2) / np.mean(logEq, axis=2) ** (7 / 2) \
           - (7 / 8) * C_tau * bval ** (3 / 2) * np.mean(logEq, axis=2) ** (-3 / 2)

    RTOP[np.abs(RTOP) == np.inf] = 0
    RTOP[np.isnan(RTOP)] = 0
    
    GFA = gfa(Eq)
    GFA[np.abs(GFA) == np.inf] = 0
    GFA[np.isnan(GFA)] = 0

    return RTOP, GFA


# -----------------------------------------------------------------------------------------------------------
def single_shell_MRI2018(data, data_b0, bval, fwhm, pred = False):
    """

    Aja-Fernández S., Pieciak T., Tristán Vega A., Vegas-Sánchez-Ferrero G., Molina V., de Luis
    Garcı́a R., Scalar diffusion-MRI measures invariant to acquisition parameters: A first step
    towards imaging biomarkers, Magnetic Resonance Imaging, vol. 54, 194-213, 2018

    """

    # prepare data

    gradients_no = data.shape[2]
    if pred:
      Eq = data
    else:
      data_b0 = data_b0[:, :, np.newaxis]
      Eq = np.divide(data, np.repeat(data_b0, gradients_no, axis=2))

    # correct for numerical instabilities
    Eq[np.isnan(Eq)] = 0
    Eq[Eq > 1] = 1
    Eq[Eq < 1e-3] = 1e-3

    # Eq filtering
    Eq2 = np.copy(Eq)

    if fwhm > 0:
        gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std

        for gg in range(0, gradients_no):
            Eq2[:, :, gg] = gaussian_filter(Eq[:, :, gg], sigma=gauss_std)

    # logarithm the data
    logEq = np.log(Eq2)
    logEq[np.isnan(logEq)] = 0

    # ASD measure (Aja-Fernández., 2018, eq. 16)
    ASD = np.mean((-logEq) ** (3 / 2), axis=2) / bval

    # CVD measure (Aja-Fernández., 2018, eq. 20)
    CVD1 = np.mean(logEq ** 2, axis=2) - np.mean(logEq, axis=2) ** 2
    CVD2 = np.mean((-logEq) ** (2), axis=2)
    CVD = np.sqrt((gradients_no / (gradients_no - 1)) * CVD1 / CVD2)

    ASD[np.abs(ASD) == np.inf] = 0
    ASD[np.isnan(ASD)] = 0

    CVD[np.abs(CVD) == np.inf] = 0
    CVD[np.isnan(CVD)] = 0

    return ASD, CVD
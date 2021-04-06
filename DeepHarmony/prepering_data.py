import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch


# DIPY library
from dipy.io import read_bvals_bvecs

# Loading dMRI data
filename_bvals =  'effective_bvalues.txt'
filename_bvecs = 'DWI_encoding_directions.txt'

# load Hansen brain data
data_path1 = 'human_brain.mat'
mat_data = sio.loadmat(data_path1)
diffusion_data = mat_data['human_brain_data']

def get_bvec_bval(filename_bvals, filename_bvecs):
    # load b-values and vectors
    bvals_original, bvecs_original = read_bvals_bvecs(filename_bvals, filename_bvecs)
    return bvecs_original, bvals_original

def select_bvals(x,y, bvecs_original, bvals_original):
    #Selecting b-values for harmonization
    b_x = bvals_original == x
    b_y = bvals_original == y
    Y_3D = diffusion_data[:,:,:, b_y]
    X_3D = diffusion_data[:,:,:, b_x]
    return X_3D, Y_3D

def slice_3D_data(X_3D, Y_3D):
    #Slicing 3D images through axial section
    img_depth = X_3D.shape[2]
    num_images = X_3D.shape[3]

    X = np.array([X_3D[:,:,i,j] for i in range(img_depth) for j in range(num_images)]).reshape([img_depth*num_images, 1, 96, 96])
    Y = np.array([Y_3D[:,:,i,j] for i in range(img_depth) for j in range(num_images)]).reshape([img_depth*num_images, 1, 96, 96])

    return X, Y

def get_dataset(x,y):
    # Function which prepares training and validation data for dMRI harmonization

    bvecs_original, bvals_original = get_bvec_bval(filename_bvals, filename_bvecs)

    # Choosing b-values for data_images (x) and layers_images(y) data 
    X_3D, Y_3D = select_bvals(x,y, bvecs_original, bvals_original)

    #Slicing 3D data into 2D one
    X, Y  = slice_3D_data(X_3D, Y_3D)

    X = torch.from_numpy(X.astype(np.float32))/255
    Y = torch.from_numpy(Y.astype(np.float32))/255

    return X, Y 


X,Y = get_dataset(1,3)
error = np.abs( X[100]-Y[100])

plt.subplot(1,3,1)
plt.imshow(X[100].reshape([96,96]), cmap = 'gray')
plt.axis('off')
plt.title("B-value = 1000")
plt.subplot(1,3,2)
plt.imshow(Y[100].reshape([96,96]), cmap = 'gray')
plt.axis('off')
plt.title("B-value = 3000")
plt.subplot(1,3,3)
plt.imshow(error.reshape([96,96]), cmap = 'gray')
plt.axis('off')
plt.title("Diffrence")

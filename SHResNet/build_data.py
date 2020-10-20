import argparse
import numpy as np
from SHBasis import SHBasis

def build_data(args):
    save_dir = '/home/kudzia/SHResNet/data/SH_'+str(args.order)+'/b_'+str(args.bx)+'_to_b_'+str(args.by)
    data_dir = 'data/'
    subject = ['125525', '200109', '599671', '660951', '859671', '122317']
    for i in range(len(subject)):
        filename_data = data_dir + subject[i] + '/T1w/Diffusion/data.nii'
        filename_bvals = data_dir + subject[i] + '/T1w/Diffusion/bvals'
        filename_bvecs = data_dir + subject[i] + '/T1w/Diffusion/bvecs'
        sh_first = SHBasis(filename_data, filename_bvals, filename_bvecs, args.order, args.bx)
        sh_second = SHBasis(filename_data, filename_bvals, filename_bvecs, args.order, args.by)
        if i == 0:
            sh_coeff_in = sh_first.get_SHCoeff()
            for i in range(1, sh_coeff_in.shape[1] - 1):
                for j in range(1, sh_coeff_in.shape[2] - 1):
                    for k in range(1, sh_coeff_in.shape[3] - 1):
                        if i == 1 and j == 1 and k == 1:
                            image = sh_coeff_in[:, :3, :3, :3].reshape((15, 3, 3, 3, 1))
                        else:
                            tmp = sh_coeff_in[:, i - 1:i + 2, j - 1:j + 2, k - 1:k + 2].reshape((15, 3, 3, 3, 1))
                            image = np.append(image, tmp, axis=4)

            image = np.transpose(image, (4, 0, 1, 2, 3))

            label = sh_second.get_SHCoeff()
            label = label[:, 1:sh_coeff_in.shape[1] - 1, 1:sh_coeff_in.shape[2] - 1, 1:sh_coeff_in.shape[3] - 1]
            label = label.reshape((label.shape[0], 1, 1, 1, -1))
            label = np.transpose(label, (4, 0, 1, 2, 3))

        else:
            sh_coeff_in = sh_first.get_SHCoeff()[:, 5, :5, :5]
            for i in range(1, sh_coeff_in.shape[1] - 1):
                for j in range(1, sh_coeff_in.shape[2] - 1):
                    for k in range(1, sh_coeff_in.shape[3] - 1):
                        if i == 1 and j == 1 and k == 1:
                            input = sh_coeff_in[:, :3, :3, :3].reshape((15, 3, 3, 3, 1))
                        else:
                            tmp = sh_coeff_in[:, i - 1:i + 2, j - 1:j + 2, k - 1:k + 2].reshape((15, 3, 3, 3, 1))
                            input = np.append(input, tmp, axis=4)

            input = np.transpose(input, (4, 0, 1, 2, 3))
            image = np.concatenate((image, input), axis=0)
            print(image.input.shape)

            lab = sh_second.get_SHCoeff()[:, 5, :5, :5]
            lab = lab[:, 1:sh_coeff_in.shape[1] - 1, 1:sh_coeff_in.shape[2] - 1, 1:sh_coeff_in.shape[3] - 1]
            lab = lab.reshape((lab.shape[0], 1, 1, 1, -1))
            lab = np.transpose(lab, (4, 0, 1, 2, 3))
            label = np.concatenate((label, lab), axis=0)
            print(label.shape)

    if len(input) > len(label):
        input = input[0: len(label)]
    else:
        label = label[0: len(input)]

    np.save(save_dir + "/image.npy", image)
    np.save(save_dir + "/label.npy", label)

def main():
    # Experiment settings
    parser = argparse.ArgumentParser(description='Building data for SHResNet')
    parser.add_argument('-o', '--order', type=int, default=4, help='SH order (default: 4)')
    parser.add_argument('-bx', '--bx', type=int, default=1000, help='bvalue(default: 1000)')
    parser.add_argument('-by', '--by', type=int, default=2000, help='bvalue(default: 2000)')
    args = parser.parse_args()
    build_data(args)

if __name__ == '__main__':
    main()
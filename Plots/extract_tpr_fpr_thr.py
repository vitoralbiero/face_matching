import numpy as np
from sklearn import metrics
import argparse
import matplotlib.pyplot as plt
from os import path, makedirs


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    idx = np.where(array == array[idx])[0][-1]

    return idx


def compute_roc(authentic_file, impostor_file):
    authentic = np.loadtxt(authentic_file, dtype=np.str)
    if np.ndim(authentic) == 1:
        authentic_score = authentic.astype(float)
    else:
        authentic_score = authentic[:, 2].astype(float)
    authentic_y = np.ones(authentic.shape[0])

    impostor = np.loadtxt(impostor_file, dtype=np.str)
    if np.ndim(impostor) == 1:
        impostor_score = impostor.astype(float)
    else:
        impostor_score = impostor[:, 2].astype(float)
    impostor_y = np.zeros(impostor.shape[0])

    y = np.concatenate([authentic_y, impostor_y])
    scores = np.concatenate([authentic_score, impostor_score])

    print(y.shape)

    return metrics.roc_curve(y, scores, drop_intermediate=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract THR/TPR/FPR')
    parser.add_argument('-authentic', '-a', help='Authentic scores.')
    parser.add_argument('-impostor', '-i', help='Impostor scores.')
    parser.add_argument('-dest', '-d', help='Folder to save the plot.')
    parser.add_argument('-name', '-n', help='Plot name (without extension).')

    args = parser.parse_args()

    fpr, tpr, thr = compute_roc(args.authentic, args.impostor)

    values = np.column_stack((thr, tpr, fpr))

    if not path.exists(args.dest):
        makedirs(args.dest)

    values_path = path.join(args.dest, args.name + '.txt')

    values_file = open(values_path, 'w')
    values_file.write('Threshold TPR FPR\n')
    np.savetxt(values_file, values, delimiter=' ', fmt='%s')
    values_file.close()

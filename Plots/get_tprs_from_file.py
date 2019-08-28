import numpy as np
import argparse


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    idx = np.where(array == array[idx])[0][-1]

    return idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot ROC Curve')
    parser.add_argument('-file', '-f', help='File with tprs and fprs')

    args = parser.parse_args()

    np.set_printoptions(suppress=True)

    fpr_tpr = np.loadtxt(args.file, dtype=np.str, delimiter=',')
    print(fpr_tpr.shape)
    fpr_tpr = fpr_tpr[1:, :].astype(float)

    tprs = [0.00001, 0.0001, 0.001, 0.01, 0.1]

    save_tprs = np.zeros(shape=(2, len(tprs)))

    for i in range(len(tprs)):
        k = find_nearest(fpr_tpr[:, 0], tprs[i])

        save_tprs[0, i] = tprs[i]
        save_tprs[1, i] = fpr_tpr[k, 1]
        print(k)

    print(save_tprs)
    np.savetxt(args.file[:-4] + '_selected.txt', save_tprs, fmt='%s')

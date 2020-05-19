import numpy as np
from scipy.stats import ks_2samp
import argparse


def load_files(file_path):
    scores = np.loadtxt(file_path, dtype=np.str)

    if np.ndim(scores) == 1:
        scores = scores.astype(float)
    elif np.ndim(scores) == 2:
        scores = scores[:, 2].astype(float)
    else:
        scores = scores[:, 2].astype(float)

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check if distributions are stat significantly different.')
    parser.add_argument('--distribution_1', '-i1', help='Authentic/impostor distribution 1.')
    parser.add_argument('--distribution_2', '-i2', help='Authentic/impostor distribution 2.')

    args = parser.parse_args()

    scores_1 = load_files(args.distribution_1)
    scores_2 = load_files(args.distribution_2)

    result = ks_2samp(scores_1, scores_2)
    print(result)

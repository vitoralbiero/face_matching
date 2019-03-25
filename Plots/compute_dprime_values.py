import matplotlib
matplotlib.use('Agg')
import numpy as np
import argparse
import matplotlib.pyplot as plt
from os import path, makedirs
import itertools


def load_files(authentic_file, impostor_file):
    authentic = np.loadtxt(authentic_file, dtype=np.str)

    if np.ndim(authentic) == 1:
        authentic_score = authentic.astype(float)
    else:
        authentic_score = authentic[:, 2].astype(float)

    impostor = np.loadtxt(impostor_file, dtype=np.str)

    if np.ndim(impostor) == 1:
        impostor_score = impostor.astype(float)
    else:
        impostor_score = impostor[:, 2].astype(float)

    return authentic_score, impostor_score


def compute_dprime(authentic_file1, impostor_file1, l1,
                   authentic_file2, impostor_file2, l2,
                   authentic_file3, impostor_file3, l3):

    authentic_score1, impostor_score1 = load_files(
        authentic_file1, impostor_file1)

    if l2 is not None:
        authentic_score2, impostor_score2 = load_files(
            authentic_file2, impostor_file2)

    if l3 is not None:
        authentic_score3, impostor_score3 = load_files(
            authentic_file3, impostor_file3)

    d_prime1 = (abs(np.mean(authentic_score1) - np.mean(impostor_score1)) /
                np.sqrt(0.5 * (np.var(authentic_score1) + np.var(impostor_score1))))

    print('d-prime for {} is: {} '.format(l1, d_prime1))

    if l2 is not None:
        d_prime2 = (abs(np.mean(authentic_score2) - np.mean(impostor_score2)) /
                    np.sqrt(0.5 * (np.var(authentic_score2) + np.var(impostor_score2))))

        print('d-prime for {} is: {} '.format(l2, d_prime2))

    if l3 is not None:
        d_prime3 = (abs(np.mean(authentic_score3) - np.mean(impostor_score3)) /
                    np.sqrt(0.5 * (np.var(authentic_score3) + np.var(impostor_score3))))

        print('d-prime for {} is: {} '.format(l3, d_prime3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Score Histogram')
    parser.add_argument('-authentic1', '-a1', help='Authentic 1 scores.')
    parser.add_argument('-impostor1', '-i1', help='Impostor 1 scores.')
    parser.add_argument('-label1', '-l1', help='Label 1.')
    parser.add_argument('-authentic2', '-a2', help='Authentic 2 scores.')
    parser.add_argument('-impostor2', '-i2', help='Impostor 2 scores.')
    parser.add_argument('-label2', '-l2', help='Label 2.')
    parser.add_argument('-authentic3', '-a3', help='Authentic 3 scores.')
    parser.add_argument('-impostor3', '-i3', help='Impostor 3 scores.')
    parser.add_argument('-label3', '-l3', help='Label 3.')

    args = parser.parse_args()

    compute_dprime(args.authentic1, args.impostor1, args.label1,
                   args.authentic2, args.impostor2, args.label2,
                   args.authentic3, args.impostor3, args.label3)

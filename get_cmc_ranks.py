import numpy as np
import argparse
import matplotlib.pyplot as plt
from os import path, makedirs


def load_files(authentic_file, impostor_file, label_file_probe, label_file_gallery):
    authentic = np.loadtxt(authentic_file, dtype=np.str)
    impostor = np.loadtxt(impostor_file, dtype=np.str)
    label_probe = np.loadtxt(label_file_probe, dtype=np.str)

    label_gallery = None

    if label_file_gallery is not None:
        label_gallery = np.loadtxt(label_file_gallery, dtype=np.str)

    return authentic, impostor, label_probe, label_gallery


def get_ranks(authentic_file, impostor_file, label_file_probe, label_file_gallery):
    ranks_save = label_file_probe[:-4] + '_ranks.txt'

    authentic, impostor, label_probe, label_gallery = load_files(
        authentic_file, impostor_file, label_file_probe, label_file_gallery)

    ranks = []

    for i in range(len(label_probe)):
        label_idx = label_probe[i, 0]

        rank = 0
        label_scores = authentic[authentic[:, 0] == label_idx][:, 2].astype(float)

        best_authentic = -float('Inf')
        # check if subject with label_idx have scores in collum 1
        if len(label_scores) > 0:
            best_authentic = np.max(label_scores)

        if label_gallery is None:
            label_scores = authentic[authentic[:, 1] == label_idx][:, 2].astype(float)

            # check if subject with label_idx have scores in collum 2 (only when probe == gallery)
            if len(label_scores) > 0:
                best_authentic = max(best_authentic, np.max(label_scores))

        impostor_scores = impostor[impostor[:, 0] == label_idx][:, 2].astype(float)
        rank = len(impostor_scores[impostor_scores > best_authentic])

        if label_gallery is None:
            impostor_scores = impostor[impostor[:, 1] == label_idx][:, 2].astype(float)

            rank += len(impostor_scores[impostor_scores > best_authentic])

        rank += 1

        ranks.append(rank)

    np.savetxt(ranks_save, ranks, delimiter=' ', fmt='%s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Ranks')
    parser.add_argument('-authentic', '-a', help='Authentic 1 scores.')
    parser.add_argument('-impostor', '-i', help='Impostor 1 scores.')
    parser.add_argument('-label_probe', '-lp', help='Label probe.')
    parser.add_argument('-label_gallery', '-lg', help='Label gallery.')

    args = parser.parse_args()

    get_ranks(args.authentic, args.impostor, args.label_probe, args.label_gallery)

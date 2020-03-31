import numpy as np
import argparse
from os import path


def load_files(authentic_file, impostor_file, label_file_probe, label_file_gallery):
    authentic = np.loadtxt(authentic_file, dtype=np.str)
    impostor = np.loadtxt(impostor_file, dtype=np.str)
    label_probe = np.loadtxt(label_file_probe, dtype=np.str)

    label_gallery = None

    if label_file_gallery is not None:
        label_gallery = np.loadtxt(label_file_gallery, dtype=np.str)

    return authentic, impostor, label_probe, label_gallery


def get_ranks(authentic_file, impostor_file, label_file_probe, label_file_gallery, distance):
    ranks_save = label_file_probe[:-4] + '_ranks.txt'
    best_auth_imp_save = label_file_probe[:-4] + '_label_auth_imp_diff.txt'

    authentic, impostor, label_probe, label_gallery = load_files(
        authentic_file, impostor_file, label_file_probe, label_file_gallery)

    ranks = []
    best_auth_imp = []

    for i in range(len(label_probe)):
        label_idx = label_probe[i, 0]

        rank = 0
        authentic_scores = authentic[authentic[:, 0] == label_idx][:, 2].astype(float)

        if distance:
            best_authentic = float('Inf')
            best_impostor = float('Inf')
        else:
            best_authentic = -float('Inf')
            best_impostor = -float('Inf')

        # check if subject with label_idx have scores in collum 1
        if len(authentic_scores) > 0:
            if distance:
                best_authentic = np.min(authentic_scores)
            else:
                best_authentic = np.max(authentic_scores)

        if label_gallery is None:
            authentic_scores = authentic[authentic[:, 1] == label_idx][:, 2].astype(float)

            # check if subject with label_idx have scores in collum 2 (only when probe == gallery)
            if len(authentic_scores) > 0:
                if distance:
                    best_authentic = min(best_authentic, np.min(authentic_scores))
                else:
                    best_authentic = max(best_authentic, np.max(authentic_scores))

        # check if label has an authentic score, otherwise skip
        if best_authentic == -float('Inf') or best_authentic == float('Inf'):
            continue

        impostor_scores = impostor[impostor[:, 0] == label_idx][:, 2].astype(float)

        if distance:
            rank = len(impostor_scores[impostor_scores < best_authentic])
            if len(impostor_scores) > 1:
                best_impostor = np.min(impostor_scores)
        else:
            rank = len(impostor_scores[impostor_scores > best_authentic])
            if len(impostor_scores) > 1:
                best_impostor = np.max(impostor_scores)

        if label_gallery is None:
            impostor_scores = impostor[impostor[:, 1] == label_idx][:, 2].astype(float)

            if distance:
                rank += len(impostor_scores[impostor_scores < best_authentic])
                if len(impostor_scores) > 1:
                    best_impostor = min(best_impostor, np.min(impostor_scores))
            else:
                rank += len(impostor_scores[impostor_scores > best_authentic])
                if len(impostor_scores) > 1:
                    best_impostor = max(best_impostor, np.max(impostor_scores))

        # check if label has an impostor score, otherwise skip
        if best_impostor == -float('Inf') or best_impostor == float('Inf'):
            continue

        rank += 1

        ranks.append(rank)
        best_auth_imp.append([label_idx, best_authentic, best_impostor,
                              round(best_authentic - best_impostor, 6)])

    best_auth_imp = np.array(best_auth_imp)
    if distance:
        best_auth_imp = best_auth_imp[best_auth_imp[:, 3].argsort()][::-1]
    else:
        best_auth_imp = best_auth_imp[best_auth_imp[:, 3].argsort()]

    np.savetxt(best_auth_imp_save, best_auth_imp, delimiter=' ', fmt='%s')
    np.savetxt(ranks_save, ranks, delimiter=' ', fmt='%s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Ranks')
    parser.add_argument('-authentic', '-a', help='Authentic 1 scores.')
    parser.add_argument('-impostor', '-i', help='Impostor 1 scores.')
    parser.add_argument('-label_probe', '-lp', help='Label probe.')
    parser.add_argument('-label_gallery', '-lg', help='Label gallery.')
    parser.add_argument('--distance', '-d', help='Distance or similarity metric.', action='store_true')

    args = parser.parse_args()
    print(args.distance)

    get_ranks(args.authentic, args.impostor, args.label_probe, args.label_gallery, args.distance)

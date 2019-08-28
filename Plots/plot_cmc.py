import numpy as np
from sklearn import metrics
import argparse
import matplotlib.pyplot as plt
from os import path, makedirs


def compute_accuracies(ranks_file, total_ranks):
    rank = np.loadtxt(ranks_file, dtype=np.float)

    if np.ndim(rank) == 1:
        rank_scores = rank.astype(float)
    else:
        rank_scores = rank[:, 2].astype(float)

    acc = np.zeros(shape=total_ranks)

    for i in range(total_ranks):
        acc[i] = len(rank_scores[rank_scores <= (i + 1)]) / len(rank_scores)

    return acc


def plot(title, total_ranks, acc1, l1, acc2, l2, acc3, l3):
    plt.rcParams["figure.figsize"] = [7, 5]
    plt.rcParams['font.size'] = 12

    plt.grid(True, zorder=0, linestyle='dashed')

    if title is not None:
        plt.title(title, y=1.08)

    if total_ranks is None:
        total_ranks = np.max(acc1)

    if args.rank2 is not None:
        total_ranks = max(total_ranks, np.max(acc2))

    if args.rank3 is not None:
        total_ranks = max(total_ranks, np.max(acc3))

    ranks = np.arange(1, total_ranks + 1)

    plt.plot(ranks, acc1, 'C1', label=l1)

    if l2 is not None:
        plt.plot(ranks, acc2, 'C0', label=l2)

    if l3 is not None:
        plt.plot(ranks, acc3, 'C3', label=l3)

    plt.legend(loc='lower right')
    plt.xlim([1, total_ranks])
    # plt.ylim([0, 1])
    plt.ylabel('Accuracy')
    plt.xlabel('Rank')

    plt.tight_layout(pad=0)

    return plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot CMC Curve')
    parser.add_argument('-total_ranks', '-r', help='Number of ranks to compute.')
    parser.add_argument('-rank1', '-r1', help='Rank scores 1.')
    parser.add_argument('-label1', '-l1', help='Label 1.')
    parser.add_argument('-rank2', '-r2', help='Rank scores 2.')
    parser.add_argument('-label2', '-l2', help='Label 2.')
    parser.add_argument('-rank3', '-r3', help='Rank scores 3.')
    parser.add_argument('-label3', '-l3', help='Label 3.')
    parser.add_argument('-title', '-t', help='Plot title.')
    parser.add_argument('-dest', '-d', help='Folder to save the plot.')
    parser.add_argument('-name', '-n', help='Plot name (without extension).')

    args = parser.parse_args()

    total_ranks = int(args.total_ranks)

    acc2 = None
    acc3 = None

    acc1 = compute_accuracies(args.rank1, total_ranks)

    if args.rank2 is not None:
        acc2 = compute_accuracies(args.rank2, total_ranks)

    if args.rank3 is not None:
        acc3 = compute_accuracies(args.rank3, total_ranks)

    plot(args.title, total_ranks, acc1, args.label1, acc2, args.label2, acc3, args.label3)

    if not path.exists(args.dest):
        makedirs(args.dest)

    plot_path = path.join(args.dest, args.name + '.png')

    plt.savefig(plot_path, dpi=600)

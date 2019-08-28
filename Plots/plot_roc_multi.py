import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn import metrics
import argparse
import matplotlib.pyplot as plt
from os import path, makedirs, listdir
import re


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    idx = np.where(array == array[idx])[0][-1]

    # idx = idx[idx.shape]

    return idx


def compute_roc(authentic_file, impostor_file, ignore_aut=-1, ignore_imp=-1):
    authentic_score = np.loadtxt(authentic_file, dtype=np.str)

    if ignore_aut != -1:
        authentic_score = authentic_score[authentic_score[:, 0].astype(int) < ignore_aut, 1].astype(float)

    elif np.ndim(authentic_score) == 1:
        authentic_score = authentic_score.astype(float)
    else:
        authentic_score = authentic_score[:, 2].astype(float)
    authentic_y = np.ones(authentic_score.shape[0])

    impostor_score = np.loadtxt(impostor_file, dtype=np.str)

    if ignore_imp != -1:
        impostor_score = impostor_score[impostor_score[:, 0].astype(int) < ignore_imp, 1].astype(float)

    elif np.ndim(impostor_score) == 1:
        impostor_score = impostor_score.astype(float)
    else:
        impostor_score = impostor_score[:, 2].astype(float)
    impostor_y = np.zeros(impostor_score.shape[0])

    y = np.concatenate([authentic_y, impostor_y])
    scores = np.concatenate([authentic_score, impostor_score])

    # invert scores in case of distance instead of similarity
    # scores *= -1

    print(y.shape)

    return metrics.roc_curve(y, scores, drop_intermediate=False)


def sorted_aphanumeric(data):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(data, key=alphanum_key)


def plot(title, alg1_path, label1, alg2_path, label2, ylim):
    plt.grid(True, zorder=0, linestyle='dashed')

    # if title is not None and title != '' and title != ' ':
    #   plt.title(title)

    plt.gca().set_xscale('log')

    begin_x = 1e-4
    end_x = 1e0
    print(begin_x, end_x)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

    i = 0
    for subset in sorted_aphanumeric(listdir(alg1_path)):
        authentic_path = path.join(alg1_path, subset, 'all_authentic.txt')
        impostor_path = path.join(alg1_path, subset, 'all_impostor.txt')

        subset_label = subset[8:] + ' ' + label1

        fpr, tpr, thr = compute_roc(authentic_path, impostor_path)
        plt.plot(fpr, tpr, colors[i], label=subset_label, linestyle='dashed')

        i += 1

    i = 0
    for subset in sorted_aphanumeric(listdir(alg2_path)):
        authentic_path = path.join(alg2_path, subset, 'all_authentic.txt')
        impostor_path = path.join(alg2_path, subset, 'all_impostor.txt')

        subset_label = subset[8:] + ' ' + label2

        fpr, tpr, thr = compute_roc(authentic_path, impostor_path)
        plt.plot(fpr, tpr, colors[i], label=subset_label)

        i += 1

    legend1 = plt.legend(loc='lower right', fontsize=12)
    plt.xlim([begin_x, end_x])
    plt.ylim([ylim, 1])
    # plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Match Rate')

    plt.tight_layout(pad=0)

    handles = []
    for c in colors:
        handles.append(Rectangle((0, 0), 1, 1, color=c, fill=True))

    handles = np.asarray(handles)

    plt.legend(handles, labels, loc="upper left", fontsize=10)
    plt.gca().add_artist(legend1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot ROC Curve')
    parser.add_argument('-alg1_path', '-alg1', help='Algorithm 1')
    parser.add_argument('-label1', '-l1', help='Algorithm 1 label')
    parser.add_argument('-alg2_path', '-alg2', help='Algorithm 2')
    parser.add_argument('-label2', '-l2', help='Algorithm 2 label')
    parser.add_argument('-title', '-t', help='Plot title.')
    parser.add_argument('-dest', '-d', help='Folder to save the plot.')
    parser.add_argument('-name', '-n', help='Plot name (without extension).')
    parser.add_argument('--ylim', default=0.75)

    args = parser.parse_args()

    plot(args.title, args.alg1_path, args.label1, args.alg2_path, args.label2, float(args.ylim))

    if not path.exists(args.dest):
        makedirs(args.dest)

    plot_path = path.join(args.dest, args.name + '.png')

    plt.savefig(plot_path, dpi=150)

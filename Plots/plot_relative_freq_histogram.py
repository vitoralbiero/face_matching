import matplotlib
matplotlib.use('Agg')
import numpy as np
import argparse
import matplotlib.pyplot as plt
from os import path, makedirs
import itertools


def load_files(authentic_file, impostor_file, ignore_aut=-1, ignore_imp=-1):
    authentic = np.loadtxt(authentic_file, dtype=np.str)

    if ignore_aut != -1:
        authentic_score = authentic[authentic[:, 0].astype(int) < ignore_aut, 1].astype(float)

    elif np.ndim(authentic) == 1:
        authentic_score = authentic.astype(float)
    else:
        authentic_score = authentic[:, 2].astype(float)

    impostor = np.loadtxt(impostor_file, dtype=np.str)

    if ignore_imp != -1:
        impostor_score = impostor[impostor[:, 0].astype(int) < ignore_imp, 1].astype(float)

    elif np.ndim(impostor) == 1:
        impostor_score = impostor.astype(float)
    else:
        impostor_score = impostor[:, 2].astype(float)

    return authentic_score, impostor_score


def plot_histogram(authentic_file1, impostor_file1, l1,
                   authentic_file2, impostor_file2, l2,
                   authentic_file3, impostor_file3, l3, title):
    authentic_score1, impostor_score1 = load_files(
        authentic_file1, impostor_file1)

    if l2 is not None:
        authentic_score2, impostor_score2 = load_files(
            authentic_file2, impostor_file2)

    if l3 is not None:
        authentic_score3, impostor_score3 = load_files(
            authentic_file3, impostor_file3)

    # bins = np.linspace(0, 1.0, 100)

    plt.rcParams["figure.figsize"] = [6, 5]
    plt.rcParams['font.size'] = 12

    color_a = 'g'
    color_i = 'r'

    if l2 is not None:
        color_a = 'C1'
        color_i = 'C1'

    if title is not None:
        plt.title(title)

    plt.hist(authentic_score1, bins='auto', histtype='step', density=True,
             label=l1 + ' Authentic', color=color_a, linewidth=1.5)
    plt.hist(impostor_score1, bins='auto', histtype='step', density=True,
             label=l1 + ' Impostor', color=color_i, linestyle='dashed',
             linewidth=1.5)

    if l2 is not None:
        plt.hist(authentic_score2, bins='auto', histtype='step', density=True,
                 label=l2 + ' Authentic', color='C0', linewidth=1.5)
        plt.hist(impostor_score2, bins='auto', histtype='step', density=True,
                 label=l2 + ' Impostor', color='C0', linestyle='dashed',
                 linewidth=1.5)

    if l3 is not None:
        plt.hist(authentic_score3, bins='auto', histtype='step', density=True,
                 label=l3 + ' Authentic', color='C3', linewidth=1.5)
        plt.hist(impostor_score3, bins='auto', histtype='step', density=True,
                 label=l3 + ' Impostor', color='C3', linestyle='dashed',
                 linewidth=1.5)

    if l3 is not None:
        ncol = 3
    elif l2 is not None:
        ncol = 2
    else:
        ncol = 2

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=0, ncol=ncol, fontsize=12, edgecolor='black', handletextpad=0.3)

    # plt.gca().invert_xaxis()
    plt.ylabel('Relative Frequency')
    plt.xlabel('Match Scores')

    plt.tight_layout(pad=0)


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
    parser.add_argument('-title', '-t', help='Plot title.')
    parser.add_argument('-dest', '-d', help='Folder to save the plot.')
    parser.add_argument('-name', '-n', help='Plot name (without extension).')

    args = parser.parse_args()

    plot_histogram(args.authentic1, args.impostor1, args.label1,
                   args.authentic2, args.impostor2, args.label2,
                   args.authentic3, args.impostor3, args.label3,
                   args.title)

    if not path.exists(args.dest):
        makedirs(args.dest)

    plot_path = path.join(args.dest, args.name + '.png')

    plt.savefig(plot_path, dpi=600)

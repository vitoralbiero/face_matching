import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn import metrics
import argparse
import matplotlib.pyplot as plt
from os import path, makedirs


def get_roc(authentic_file, impostor_file):
    authentic_score = np.loadtxt(authentic_file, dtype=np.str)

    if np.ndim(authentic_score) == 1:
        authentic_score = authentic_score.astype(float)
    else:
        authentic_score = authentic_score[:, 2].astype(float)

    authentic_y = np.ones(authentic_score.shape[0])

    impostor_score = np.loadtxt(impostor_file, dtype=np.str)

    if np.ndim(impostor_score) == 1:
        impostor_score = impostor_score.astype(float)
    else:
        impostor_score = impostor_score[:, 2].astype(float)
    impostor_y = np.zeros(impostor_score.shape[0])

    y = np.concatenate([authentic_y, impostor_y])
    scores = np.concatenate([authentic_score, impostor_score])

    return metrics.roc_curve(y, scores, drop_intermediate=True)


def compute_roc(authentic_file, impostor_file):
    fprs = []
    tprs = []
    thrs = []

    fpr, tpr, thr = get_roc(authentic_file + '_auc_5.txt', impostor_file + '_auc_5.txt')
    fprs.append(fpr)
    tprs.append(tpr)
    thrs.append(thr)

    fpr, tpr, thr = get_roc(authentic_file + '_auc_median.txt', impostor_file + '_auc_median.txt')
    fprs.append(fpr)
    tprs.append(tpr)
    thrs.append(thr)

    fpr, tpr, thr = get_roc(authentic_file + '_auc_95.txt', impostor_file + '_auc_95.txt')
    fprs.append(fpr)
    tprs.append(tpr)
    thrs.append(thr)

    return fprs, tprs, thrs


def plot(title, fpr1, tpr1, thr1, l1, fpr2, tpr2, thr2, l2,
         fpr3, tpr3, thr3, l3, fpr4, tpr4, thr4, l4):
    plt.rcParams["figure.figsize"] = [6, 4.5]
    plt.rcParams['font.size'] = 12

    plt.grid(True, zorder=0, linestyle='dashed')
    plt.gca().set_xscale('log')

    begin_x = 1e-5
    end_x = 1e0
    print(begin_x, end_x)

    plt.plot(fpr1[1], tpr1[1], 'C1', label=l1)
    # plt.plot(fpr1[2], tpr1[2], 'C1')
    plt.fill(np.append(fpr1[0], fpr1[2][::-1]), np.append(tpr1[0], tpr1[2][::-1]), facecolor='C1', alpha=0.5)

    if l2 is not None:
        plt.plot(fpr2[1], tpr2[1], 'C0', label=l2)
        # plt.plot(fpr2[2], tpr2[2], 'C0')
        plt.fill(np.append(fpr2[0], fpr2[2][::-1]), np.append(tpr2[0], tpr2[2][::-1]), facecolor='C0', alpha=0.5)

    if l3 is not None:
        plt.plot(fpr3[1], tpr3[1], 'C3', label=l3)
        # plt.plot(fpr3[2], tpr3[2], 'C3')
        plt.fill(np.append(fpr3[0], fpr3[2][::-1]), np.append(tpr3[0], tpr3[2][::-1]), facecolor='C3', alpha=0.5)

    if l4 is not None:
        plt.plot(fpr4[1], tpr4[1], 'C4', label=l4)
        plt.fill(np.append(fpr4[0], fpr4[2][::-1]), np.append(tpr4[0], tpr4[2][::-1]), facecolor='C4', alpha=0.5)

    plt.legend(loc='lower right', fontsize=12)
    plt.xlim([begin_x, end_x])
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.ylim([0.7, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Match Rate')

    plt.tight_layout(pad=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot ROC Curve')
    parser.add_argument('-authentic1', '-a1', help='Authentic scores 1.')
    parser.add_argument('-impostor1', '-i1', help='Impostor scores 1.')
    parser.add_argument('-label1', '-l1', help='Label 1.')
    parser.add_argument('-authentic2', '-a2', help='Authentic scores 2.')
    parser.add_argument('-impostor2', '-i2', help='Impostor scores 2.')
    parser.add_argument('-label2', '-l2', help='Label 2.')
    parser.add_argument('-authentic3', '-a3', help='Authentic scores 3.')
    parser.add_argument('-impostor3', '-i3', help='Impostor scores 3.')
    parser.add_argument('-label3', '-l3', help='Label 3.')
    parser.add_argument('-authentic4', '-a4', help='Authentic scores 4.')
    parser.add_argument('-impostor4', '-i4', help='Impostor scores 4.')
    parser.add_argument('-label4', '-l4', help='Label 4.')
    parser.add_argument('-title', '-t', help='Plot title.')
    parser.add_argument('-dest', '-d', help='Folder to save the plot.')
    parser.add_argument('-name', '-n', help='Plot name (without extension).')

    args = parser.parse_args()

    fpr2, tpr2, thr2 = (None, None, None)
    fpr3, tpr3, thr3 = (None, None, None)
    fpr4, tpr4, thr4 = (None, None, None)

    fpr1, tpr1, thr1 = compute_roc(args.authentic1, args.impostor1)

    if args.authentic2 is not None:
        fpr2, tpr2, thr2 = compute_roc(args.authentic2, args.impostor2)

    if args.authentic3 is not None:
        fpr3, tpr3, thr3 = compute_roc(args.authentic3, args.impostor3)

    if args.authentic4 is not None:
        fpr4, tpr4, thr4 = compute_roc(args.authentic4, args.impostor4)

    plot(args.title, fpr1, tpr1, thr1, args.label1,
         fpr2, tpr2, thr2, args.label2, fpr3, tpr3, thr3, args.label3,
         fpr4, tpr4, thr4, args.label4)

    if not path.exists(args.dest):
        makedirs(args.dest)

    plot_path = path.join(args.dest, args.name + '.png')
plt.savefig(plot_path, dpi=150)

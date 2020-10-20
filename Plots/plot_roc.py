import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn import metrics
import argparse
import matplotlib.pyplot as plt
from os import path, makedirs


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    idx = np.where(array == array[idx])[0][-1]

    # idx = idx[idx.shape]

    return idx


def compute_roc(authentic_file, impostor_file, ignore_aut=-1, ignore_imp=-1):
    authentic_score = np.load(authentic_file)

    if ignore_aut != -1:
        authentic_score = authentic_score[authentic_score[:, 0].astype(int) < ignore_aut, 1].astype(float)

    elif np.ndim(authentic_score) == 1:
        authentic_score = authentic_score.astype(float)
    else:
        authentic_score = authentic_score[:, 2].astype(float)
    authentic_y = np.ones(authentic_score.shape[0])

    impostor_score = np.load(impostor_file)

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


def plot(title, fpr1, tpr1, thr1, l1, fpr2, tpr2, thr2, l2,
         fpr3, tpr3, thr3, l3, fpr4, tpr4, thr4, l4, ylim):
    label_kwargs1 = {}
    label_kwargs1['bbox'] = dict(
        boxstyle='round,pad=0.5', fc='C1', alpha=0.5,
    )

    label_kwargs2 = {}
    label_kwargs2['bbox'] = dict(
        boxstyle='round,pad=0.5', fc='C0', alpha=0.5,
    )

    label_kwargs3 = {}
    label_kwargs3['bbox'] = dict(
        boxstyle='round,pad=0.5', fc='C3', alpha=0.5,
    )

    label_kwargs4 = {}
    label_kwargs4['bbox'] = dict(
        boxstyle='round,pad=0.5', fc='C4', alpha=0.5,
    )

    plt.rcParams["figure.figsize"] = [6, 5]
    plt.rcParams['font.size'] = 12

    plt.grid(True, zorder=0, linestyle='dashed')

    # if title is not None and title != '' and title != ' ':
    #   plt.title(title)

    plt.gca().set_xscale('log')

    begin_x = 1e-5
    end_x = 1e0
    print(begin_x, end_x)

    # range_f = end_x - begin_x

    # auc1 = metrics.auc(fpr1[(fpr1 >= begin_x) & (fpr1 <= end_x)], tpr1[(fpr1 >= begin_x) & (fpr1 <= end_x)])
    # auc1_per = auc1 / range_f
    # print(auc1_per)

    plt.plot(fpr1, tpr1, 'C1', label=l1)
    # + ' - AUC ({:0.5f})'.format(auc1_per))

    if l2 is not None:
        # auc2 = metrics.auc(fpr2[(fpr2 >= begin_x) & (fpr2 <= end_x)], tpr2[(fpr2 >= begin_x) & (fpr2 <= end_x)])
        # auc2_per = auc2 / range_f
        # print(auc2_per)

        plt.plot(fpr2, tpr2, 'C0', label=l2)
        # + ' - AUC ({:0.5f})'.format(auc2_per))

    if l3 is not None:
        # auc3 = metrics.auc(fpr3[(fpr3 >= begin_x) & (fpr3 <= end_x)], tpr3[(fpr3 >= begin_x) & (fpr3 <= end_x)])
        # auc3_per = auc3 / range_f
        # print(auc3_per)

        plt.plot(fpr3, tpr3, 'C3', label=l3)
        # + ' - AUC ({:0.5f})'.format(auc3_per))

    if l4 is not None:
        # auc3 = metrics.auc(fpr3[(fpr3 >= begin_x) & (fpr3 <= end_x)], tpr3[(fpr3 >= begin_x) & (fpr3 <= end_x)])
        # auc3_per = auc3 / range_f
        # print(auc3_per)

        plt.plot(fpr4, tpr4, 'C4', label=l4)

    thrs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]

    if l2 is not None:
        if l3 is not None:
            if l4 is not None:
                save_tprs = np.zeros(shape=(5, len(thrs)))
            else:
                save_tprs = np.zeros(shape=(4, len(thrs)))
        else:
            save_tprs = np.zeros(shape=(3, len(thrs)))
    else:
        save_tprs = np.zeros(shape=(2, len(thrs)))

    offset = 1

    invert = 1

    for i in range(len(thrs)):
        y_pos = -30

        save_tprs[0, i] = thrs[i]

        if thrs[i] != 0.00001:
            offset = 0

        k = find_nearest(fpr1, thrs[i])
        t1 = str(np.round(thr1[k], 2) * invert)
        x1 = fpr1[k + offset]
        y1 = tpr1[k]

        save_tprs[1, i] = y1

        if l2 is not None:
            k = find_nearest(fpr2, thrs[i])
            t2 = str(np.round(thr2[k], 2) * invert)
            x2 = fpr2[k + offset]
            y2 = tpr2[k]

            save_tprs[2, i] = y2

        if l3 is not None:
            k = find_nearest(fpr3, thrs[i])
            t3 = str(np.round(thr3[k], 2) * invert)
            x3 = fpr3[k + offset]
            y3 = tpr3[k]

            save_tprs[3, i] = y3

        if l4 is not None:
            k = find_nearest(fpr4, thrs[i])
            t4 = str(np.round(thr4[k], 2) * invert)
            x4 = fpr4[k + offset]
            y4 = tpr4[k]

            save_tprs[4, i] = y4

        # y_pos = -55

        plt.annotate(t1, (x1, y1), xycoords='data',
                     xytext=(15, y_pos), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->"), fontsize=10,
                     **label_kwargs1)

        if l2 is not None:
            if abs(y1 - y2) < 0.02:
                y_pos -= 30

            # y_pos = -20

            plt.annotate(t2, (x2, y2), xycoords='data',
                         xytext=(15, y_pos), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->"), fontsize=10,
                         **label_kwargs2)

        if l3 is not None:
            y_pos = -30

            if (abs(y1 - y2) < 0.02) and (abs(y2 - y3) < 0.05):
                y_pos -= 60
            elif (abs(y2 - y3) < 0.02):
                y_pos -= 30

            # if i == 0.00001:
            #    y_pos = -25
            # else:
            #    y_pos = -40

            plt.annotate(t3, (x3, y3), xycoords='data',
                         xytext=(15, y_pos), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->"), fontsize=10,
                         **label_kwargs3)

        if l4 is not None:
            y_pos = -30

            if (abs(y2 - y3) < 0.02) and (abs(y3 - y4) < 0.05):
                y_pos -= 60
            elif (abs(y3 - y4) < 0.02):
                y_pos -= 30

            # if i == 0.00001:
            #    y_pos = -25
            # else:
            #    y_pos = -40

            plt.annotate(t4, (x4, y4), xycoords='data',
                         xytext=(15, y_pos), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->"), fontsize=10,
                         **label_kwargs4)

    plt.legend(loc='lower right', fontsize=12)
    plt.xlim([begin_x, end_x])
    plt.ylim([ylim, 1])
    # plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Match Rate')

    plt.tight_layout(pad=0.2)

    return save_tprs


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
    parser.add_argument('--ylim', default=0.7)

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

    save_tprs = plot(args.title, fpr1, tpr1, thr1, args.label1,
                     fpr2, tpr2, thr2, args.label2, fpr3, tpr3, thr3, args.label3,
                     fpr4, tpr4, thr4, args.label4, float(args.ylim))

    if not path.exists(args.dest):
        makedirs(args.dest)

    plot_path = path.join(args.dest, args.name + '.png')
    tprs_path = path.join(args.dest, args.name + '.txt')

    np.savetxt(tprs_path, save_tprs, delimiter=' ', fmt='%s')

    plt.savefig(plot_path, dpi=150)

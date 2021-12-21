import matplotlib

matplotlib.use("Agg")
import argparse
import itertools
from os import makedirs, path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import ks_2samp


def load_files(authentic_file, impostor_file, ignore_aut=-1, ignore_imp=-1):
    print(f"Loading authentic from {authentic_file}")
    if authentic_file[-4:] == ".txt":
        authentic = np.loadtxt(authentic_file, dtype=np.str, delimiter=",")
        print(f"Converting authentic to npy")
        np.save(authentic_file[:-4] + ".npy", authentic)
    else:
        authentic = np.load(authentic_file)

    if ignore_aut != -1:
        authentic_score = authentic[authentic[:, 0].astype(int) < ignore_aut, 1].astype(
            float
        )

    elif np.ndim(authentic) == 1:
        authentic_score = authentic.astype(float)
    elif np.ndim(authentic) == 2:
        authentic_score = authentic[:, 2].astype(float)
    else:
        authentic_score = authentic[:, 2].astype(float)

    print(f"Loading impostor from {impostor_file}")
    if impostor_file[-4:] == ".txt":
        impostor = np.loadtxt(impostor_file, dtype=np.str, delimiter=",")
        print(f"Converting impostor to npy")
        np.save(impostor_file[:-4] + ".npy", impostor)
    else:
        impostor = np.load(impostor_file)

    if ignore_imp != -1:
        impostor_score = impostor[impostor[:, 0].astype(int) < ignore_imp, 1].astype(
            float
        )

    elif np.ndim(impostor) == 1:
        impostor_score = impostor.astype(float)
    elif np.ndim(impostor) == 2:
        impostor_score = impostor[:, 2].astype(float)
    else:
        impostor_score = impostor[:, 2].astype(float)

    return authentic_score, impostor_score


def plot_histogram(
    authentic_file1,
    impostor_file1,
    l1,
    authentic_file2,
    impostor_file2,
    l2,
    authentic_file3,
    impostor_file3,
    l3,
    authentic_file4,
    impostor_file4,
    l4,
    title,
):
    authentic_score1, impostor_score1 = load_files(authentic_file1, impostor_file1)

    if l2 is not None:
        authentic_score2, impostor_score2 = load_files(authentic_file2, impostor_file2)

    if l3 is not None:
        authentic_score3, impostor_score3 = load_files(authentic_file3, impostor_file3)

    if l4 is not None:
        authentic_score4, impostor_score4 = load_files(authentic_file4, impostor_file4)

    plt.rcParams["figure.figsize"] = [6, 4.5]
    plt.rcParams["font.size"] = 12

    color_a = "b"
    color_b = "r"

    if l3 is not None:
        color_a = "C0"
        color_b = "C1"

    if title is not None:
        plt.title(title, y=1.16)

    plt.hist(
        authentic_score1,
        bins="auto",
        histtype="step",
        density=True,
        label=l1 + " Authentic",
        color=color_a,
        linewidth=1.5,
    )
    plt.hist(
        impostor_score1,
        bins="auto",
        histtype="step",
        density=True,
        label=l1 + " Impostor",
        color=color_a,
        linestyle="dashed",
        linewidth=1.5,
    )

    if l2 is not None:
        plt.hist(
            authentic_score2,
            bins="auto",
            histtype="step",
            density=True,
            label=l2 + " Authentic",
            color=color_b,
            linewidth=1.5,
        )
        plt.hist(
            impostor_score2,
            bins="auto",
            histtype="step",
            density=True,
            label=l2 + " Impostor",
            color=color_b,
            linestyle="dashed",
            linewidth=1.5,
        )

    if l3 is not None:
        plt.hist(
            authentic_score3,
            bins="auto",
            histtype="step",
            density=True,
            label=l3 + " Authentic",
            color="C3",
            linewidth=1.5,
        )
        plt.hist(
            impostor_score3,
            bins="auto",
            histtype="step",
            density=True,
            label=l3 + " Impostor",
            color="C3",
            linestyle="dashed",
            linewidth=1.5,
        )

    if l4 is not None:
        plt.hist(
            authentic_score4,
            bins="auto",
            histtype="step",
            density=True,
            label=l4 + " Authentic",
            color="C4",
            linewidth=1.5,
        )
        plt.hist(
            impostor_score4,
            bins="auto",
            histtype="step",
            density=True,
            label=l4 + " Impostor",
            color="C4",
            linestyle="dashed",
            linewidth=1.5,
        )

    if l3 is not None:
        ncol = 2
    elif l2 is not None:
        ncol = 2
    else:
        ncol = 1

    legend1 = plt.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=ncol,
        fontsize=10,
        edgecolor="black",
        handletextpad=0.3,
    )

    plt.ylabel("Relative Frequency")
    plt.xlabel("Match Scores")
    
    plt.tight_layout(pad=0.2)

    d_prime1 = abs(np.mean(authentic_score1) - np.mean(impostor_score1)) / np.sqrt(
        0.5 * (np.var(authentic_score1) + np.var(impostor_score1))
    )

    save_file = open(path.join(args.dest, args.name + ".txt"), "w")

    print("d-prime for {} is: {} ".format(l1, d_prime1))
    save_file.write("d-prime for {} is: {} ".format(l1, d_prime1))
    colors = []
    labels = []

    colors.append(color_a)
    labels.append("d-prime: {}".format(np.round(d_prime1, 3)))
    save_file.write("d-prime: {}".format(np.round(d_prime1, 3)))

    if l2 is not None:
        d_prime2 = abs(np.mean(authentic_score2) - np.mean(impostor_score2)) / np.sqrt(
            0.5 * (np.var(authentic_score2) + np.var(impostor_score2))
        )
        colors.append(color_b)
        labels.append("d-prime: {}".format(np.round(d_prime2, 3)))

        print("d-prime for {} is: {} ".format(l2, d_prime2))
        save_file.write("d-prime for {} is: {} ".format(l2, d_prime2))

    if l3 is not None:
        d_prime3 = abs(np.mean(authentic_score3) - np.mean(impostor_score3)) / np.sqrt(
            0.5 * (np.var(authentic_score3) + np.var(impostor_score3))
        )
        colors.append("C3")
        labels.append("d-prime: {}".format(np.round(d_prime3, 3)))

        print("d-prime for {} is: {} ".format(l3, d_prime3))
        save_file.write("d-prime for {} is: {} ".format(l3, d_prime3))

    if l4 is not None:
        d_prime4 = abs(np.mean(authentic_score4) - np.mean(impostor_score4)) / np.sqrt(
            0.5 * (np.var(authentic_score4) + np.var(impostor_score4))
        )
        colors.append("C4")
        labels.append("d-prime: {}".format(np.round(d_prime4, 3)))

        print("d-prime for {} is: {} ".format(l4, d_prime4))
        save_file.write("d-prime: {}".format(np.round(d_prime4, 3)))

    colors = np.asarray(colors)
    labels = np.asarray(labels)

    handles = []
    for c in colors:
        handles.append(Rectangle((0, 0), 1, 1, color=c, fill=True))

    handles = np.asarray(handles)

    plt.legend(handles, labels, loc="upper left", fontsize=10)
    plt.gca().add_artist(legend1)

    plot_path = path.join(args.dest, args.name + ".png")
    plt.savefig(plot_path, dpi=150)

    if l2 is not None:
        result = ks_2samp(authentic_score1, authentic_score2)
        print(f"{l1} and {l2} authentic KS test: {result}")
        save_file.write(f"{l1} and {l2} authentic KS test: {result}")

        d_prime = abs(np.mean(authentic_score1) - np.mean(authentic_score2)) / np.sqrt(
            0.5 * (np.var(authentic_score1) + np.var(authentic_score2))
        )
        print(f"{l1} and {l2} authentic d-prime: {d_prime}")
        save_file.write(f"{l1} and {l2} authentic d-prime: {d_prime}")

        result = ks_2samp(impostor_score1, impostor_score2)
        print(f"{l1} and {l2} impostor KS test: {result}")
        save_file.write(f"{l1} and {l2} impostor KS test: {result}")

        d_prime = abs(np.mean(impostor_score1) - np.mean(impostor_score2)) / np.sqrt(
            0.5 * (np.var(impostor_score1) + np.var(impostor_score2))
        )
        print(f"{l1} and {l2} impostor d-prime: {d_prime}")
        save_file.write(f"{l1} and {l2} impostor d-prime: {d_prime}")

    save_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Score Histogram")
    parser.add_argument("-authentic1", "-a1", help="Authentic 1 scores.")
    parser.add_argument("-impostor1", "-i1", help="Impostor 1 scores.")
    parser.add_argument("-label1", "-l1", help="Label 1.")
    parser.add_argument("-authentic2", "-a2", help="Authentic 2 scores.")
    parser.add_argument("-impostor2", "-i2", help="Impostor 2 scores.")
    parser.add_argument("-label2", "-l2", help="Label 2.")
    parser.add_argument("-authentic3", "-a3", help="Authentic 3 scores.")
    parser.add_argument("-impostor3", "-i3", help="Impostor 3 scores.")
    parser.add_argument("-label3", "-l3", help="Label 3.")
    parser.add_argument("-authentic4", "-a4", help="Authentic 4 scores.")
    parser.add_argument("-impostor4", "-i4", help="Impostor 4 scores.")
    parser.add_argument("-label4", "-l4", help="Label 4.")
    parser.add_argument("-title", "-t", help="Plot title.")
    parser.add_argument("-dest", "-d", help="Folder to save the plot.")
    parser.add_argument("-name", "-n", help="Plot name (without extension).")

    args = parser.parse_args()

    if not path.exists(args.dest):
        makedirs(args.dest)

    plot_histogram(
        args.authentic1,
        args.impostor1,
        args.label1,
        args.authentic2,
        args.impostor2,
        args.label2,
        args.authentic3,
        args.impostor3,
        args.label3,
        args.authentic4,
        args.impostor4,
        args.label4,
        args.title,
    )

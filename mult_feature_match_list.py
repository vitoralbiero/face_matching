import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from os import path, makedirs
from multiprocessing import Pool
import os
from scipy.spatial import distance


PROBE_FILE = None
PROBE = None
GALLERY_FILE = None
GALLERY = None
TWINS = None
ID_SIZE = None
DATASET = None
METRIC = None


def match_features(output, group):
    authentic_save = path.join(output, '{}_authentic.txt'.format(group))
    impostor_save = path.join(output, '{}_impostor.txt'.format(group))
    twins_save = path.join(output, '{}_twins.txt'.format(group))
    labels_save = path.join(output, '{}_labels.txt'.format(group))

    # run this in multiple processes to speed things up
    pool = Pool(os.cpu_count())
    print(os.cpu_count())

    impostor_file = open(impostor_save, 'w')
    authentic_file = open(authentic_save, 'w')
    labels_file = []

    if DATASET == 'ND':
        twins_file = open(twins_save, 'w')

    for authentic, impostor, twins, label in pool.imap_unordered(match, PROBE):
        if impostor.shape[0] > 0:
            np.savetxt(impostor_file, impostor, delimiter=' ', fmt='%i %i %s')

        if authentic.shape[0] > 0:
            np.savetxt(authentic_file, authentic, delimiter=' ', fmt='%i %i %s')

        if twins.shape[0] > 0:
            np.savetxt(twins_file, twins, delimiter=' ', fmt='%i %i %s')

        if label is not None:
            labels_file.append(label)

    if GALLERY_FILE != PROBE_FILE:
        labels_gallery = path.join(output, '{}_labels_gallery.txt'.format(group))
        labels_gallery_file = []

        for j in range(len(GALLERY)):
            image_b_path = GALLERY[j]
            image_b = path.split(image_b_path)[1]
            label = (j, image_b[:-4])
            labels_gallery_file.append(label)

        np.savetxt(labels_gallery, labels_gallery_file, delimiter=' ', fmt='%s')

    impostor_file.close()
    authentic_file.close()
    labels_file = np.array(labels_file)
    np.savetxt(labels_save, labels_file, delimiter=' ', fmt='%s')

    if DATASET == 'ND':
        twins_file.close()


def chisquare(p, q):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    bin_dists = (p - q)**2 / (p + q + np.finfo('float').eps)
    return np.sum(bin_dists)


def match(probe):
    authentic_list = []
    impostor_list = []
    twins_list = []

    image_a_path = probe

    image_a = path.split(image_a_path)[1]
    features_a = np.load(image_a_path)

    if np.ndim(features_a) == 1:
        features_a = features_a[np.newaxis, :]

    i = np.int(np.where(PROBE == image_a_path)[0])

    label = (i, image_a[:-4])

    start = i

    if GALLERY_FILE != PROBE_FILE:
        start = -1

    for j in range(start + 1, len(GALLERY)):
        image_b_path = GALLERY[j]
        image_b = path.split(image_b_path)[1]

        if image_a == image_b:
            continue

        features_b = np.load(image_b_path)

        if np.ndim(features_b) == 1:
            features_b = features_b[np.newaxis, :]

        if METRIC == 1:
            score = np.mean(cosine_similarity(features_a, features_b))
        elif METRIC == 2:
            score = distance.euclidean(features_a, features_b)
        else:
            score = chisquare(features_a, features_b)

        comparison = (i, j, score)

        if DATASET == 'CHIYA':
            image_a_label = image_a[:-5]
            image_b_label = image_b[:-5]

        elif ID_SIZE > 0:
            image_a_label = image_a[:ID_SIZE]
            image_b_label = image_b[:ID_SIZE]
        else:
            image_a_label = image_a.split('_')[0]
            image_b_label = image_b.split('_')[0]

        if image_a_label == image_b_label:
            authentic_list.append(comparison)

        elif DATASET == 'ND':
            i_a, j_a = np.where(TWINS == image_a[:ID_SIZE])
            i_b, j_b = np.where(TWINS == image_b[:ID_SIZE])

            if i_a >= 0 and i_a == i_b:
                twins_list.append(comparison)
            else:
                impostor_list.append(comparison)
        else:
            impostor_list.append(comparison)

    impostor_list = np.round(np.array(impostor_list), 6)
    authentic_list = np.round(np.array(authentic_list), 6)
    twins_list = np.round(np.array(twins_list), 6)

    return authentic_list, impostor_list, twins_list, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match Extracted Features')
    parser.add_argument('-probe', '-p', help='Probe image list.')
    parser.add_argument('-gallery', '-g', help='Gallery image list.')
    parser.add_argument('-output', '-o', help='Output folder.')
    parser.add_argument('-dataset', '-d', help='Dataset name.')
    parser.add_argument('-group', '-gr', help='Group name, e.g. AA')
    parser.add_argument('-metric', '-m', default=1,
                        help='Metric to us: (1) Cosine Similarity; (2) Euclidean Distance; (3) Chi Square')

    args = parser.parse_args()

    if args.gallery is None:
        args.gallery = args.probe

    if not path.exists(args.output):
        makedirs(args.output)

    DATASET = args.dataset.upper()
    METRIC = int(args.metric)

    if DATASET == 'ND':
        TWINS = np.loadtxt(
            '/afs/crc.nd.edu/user/v/valbiero/ND_Dataset/' +
            'Metadata/twins.txt', delimiter=' ', dtype=np.str)
        ID_SIZE = 9
    elif DATASET == 'MORPH':
        ID_SIZE = 6
    elif DATASET == 'IJBB':
        ID_SIZE = -1
    elif DATASET == 'CHIYA':
        ID_SIZE = -1
    else:
        raise Exception('NO FILE PATTERN FOR THE DATASET INFORMED.')

    PROBE_FILE = args.probe
    PROBE = np.sort(np.loadtxt(PROBE_FILE, dtype=np.str))

    GALLERY_FILE = args.gallery
    GALLERY = np.sort(np.loadtxt(args.gallery, dtype=np.str))

    match_features(args.output, args.group)

    print(PROBE_FILE)
    print(GALLERY_FILE)

from skimage.feature import local_binary_pattern
import numpy as np
import itertools
import argparse
from os import path, listdir, makedirs
import cv2
from tantriggs_preprocessing import TanTriggsPreprocessing
import PIL
from PIL import Image


def extract_hist(img):
    hists = []
    num_points = 8
    radii = [1, 2]
    grid_x = 9
    grid_y = 9

    for radius in radii:
        lbp = local_binary_pattern(img,
                                   num_points,
                                   radius, 'nri_uniform')

        height = lbp.shape[0] // grid_x
        width = lbp.shape[1] // grid_y

        indices = itertools.product(range(int(grid_x)),
                                    range(int(grid_y)))

        for (i, j) in indices:
            top = i * height
            left = j * width
            bottom = top + height
            right = left + width

            region = lbp[top:bottom, left:right]

            n_bins = int(lbp.max() + 1)

            hist, _ = np.histogram(region, density=True,
                                   bins=n_bins,
                                   range=(0, n_bins))
            hists.append(hist)

    hists = np.asarray(hists)

    return np.ravel(hists)


def minmax(X, low, high, minX=None, maxX=None, dtype=np.float):
    X = np.asarray(X)
    if minX is None:
        minX = np.min(X)
    if maxX is None:
        maxX = np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high - low)
    X = X + low
    return np.asarray(X, dtype=dtype)


def extract_features(source, destination):
    if path.isfile(source):
        full_path = True
        source_list = np.sort(np.loadtxt(source, dtype=np.str))
    else:
        full_path = False
        source_list = listdir(source)

    tantriggs = TanTriggsPreprocessing()

    for image_name in source_list:
        if not full_path:
            image_path = path.join(source, image_name)
        else:
            image_path = image_name
            image_name = path.split(image_name)[1]

        img = Image.open(image_path)
        img = img.convert("L")
        img = img.resize((108, 108), PIL.Image.ANTIALIAS)
        img = np.array(img, dtype=np.uint8)
        img = tantriggs.extract(img)
        img = minmax(img, 0, 255)

        features = extract_hist(img)

        dest_path = destination

        if full_path:
            sub_folder = path.basename(
                path.normpath(path.split(image_path)[0]))

            dest_path = path.join(destination, sub_folder)

            if not path.exists(dest_path):
                makedirs(dest_path)

        features_name = path.join(dest_path, image_name[:-3] + 'npy')

        np.save(features_name, features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features with LBP')
    parser.add_argument('--source', '-s', help='Folder with images.')
    parser.add_argument('--dest', '-d', help='Folder to save the extractions.')

    args = parser.parse_args()

    if not path.exists(args.dest):
        makedirs(args.dest)

    extract_features(args.source, args.dest)

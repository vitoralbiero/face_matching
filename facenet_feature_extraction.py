'''
Uses weights and models implementation' from
https://github.com/nyoki-mtl/keras-facenet
'''

import numpy as np
from keras.preprocessing import image
import argparse
from os import path, listdir, makedirs
from models import facenet


def preprocess_by_img(img):
    mean = np.mean(img)
    std = np.std(img)
    height, widht = img.size
    size = height * widht
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))

    return np.multiply(np.subtract(img, mean), 1 / std_adj)


def preprocess_fixed(img):
    img -= 127.5
    img /= 128

    return img


def extract_features(weights, norm_type, source, destination):
    if path.isfile(source):
        full_path = True
        source_list = np.sort(np.loadtxt(source, dtype=np.str))
    else:
        full_path = False
        source_list = listdir(source)

    n_features = 128

    # VGGFace2 and Casia has 512 features as output
    if norm_type == 2:
        n_features = 512

    model = facenet.InceptionResNetV1(weights_path=weights, classes=n_features)

    for image_name in source_list:
        if not full_path:
            image_path = path.join(source, image_name)
        else:
            image_path = image_name
            image_name = path.split(image_name)[1]

        img = image.load_img(image_path, target_size=(160, 160))

        # this normalization is used with Celeb dataset
        if norm_type == 1:
            img = preprocess_by_img(img)

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # this normalization is used with VGGFace2 and Casia-WebFace datasets
        if norm_type == 2:
            img = preprocess_fixed(img)

        features = model.predict(img)

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
    parser = argparse.ArgumentParser(description='Extract Features with CNN')
    parser.add_argument('--weights', '-w', help='Weight path for the FaceNet.',
        default='/afs/crc.nd.edu/user/v/valbiero/Code/Features/weights/facenet_keras_ms1_celeb_weights.h5')
    parser.add_argument('--norm_type', '-n',
                        help='Type of normalization: (1) per image; (2) fixed',
                        default=1)
    parser.add_argument('--source', '-s', help='Folder with images.')
    parser.add_argument('--dest', '-d', help='Folder to save the extractions.')

    args = parser.parse_args()

    if not path.exists(args.dest):
        makedirs(args.dest)

    extract_features(args.weights, int(args.norm_type), args.source, args.dest)

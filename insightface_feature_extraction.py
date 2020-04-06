'''
Uses weights and models implementation' from
https://github.com/deepinsight/insightface
'''

import numpy as np
import argparse
from os import path, listdir, makedirs
import cv2
import sys
sys.path.insert(0, '../../insightface/deploy/')
import face_model


def extract_features(model, source, destination, weights=None):
    if path.isfile(source):
        full_path = True
        source_list = np.sort(np.loadtxt(source, dtype=np.str))
    else:
        full_path = False
        source_list = listdir(source)

    for image_name in source_list:
        if not full_path:
            image_path = path.join(source, image_name)
        else:
            image_path = image_name
            image_name = path.split(image_name)[1]

        if not image_path.lower().endswith('.png') and not image_path.lower().endswith('.jpg') \
           and not image_path.lower().endswith('.bmp'):
            continue

        dest_path = destination

        if full_path:
            sub_folder = path.basename(path.normpath(path.split(image_path)[0]))

            dest_path = path.join(destination, sub_folder)

            if not path.exists(dest_path):
                makedirs(dest_path)

        features_name = path.join(dest_path, image_name[:-3] + 'npy')

        # if path.isfile(features_name):
        #    print('Skipping...')
        #    continue

        # print(image_path)

        img = cv2.imread(image_path)
        img = model.get_input(img)
        features = model.get_feature(img)

        np.save(features_name, features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features with CNN')
    parser.add_argument('--source', '-s', help='Folder with images.')
    parser.add_argument('--dest', '-d', help='Folder to save the extractions.')

    # ArcFace params
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', help='path to model.', default='../../insightface/models/model-r100-ii/model,0')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gender_model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=1, type=int, help='mtcnn: 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

    args = parser.parse_args()

    model = face_model.FaceModel(args)

    if not path.exists(args.dest):
        makedirs(args.dest)

    extract_features(model, args.source, args.dest)

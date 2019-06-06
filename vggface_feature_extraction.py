'''
Uses weights and models implementation' from
https://github.com/rcmalli/keras-vggface
'''

from keras.engine import Model
import numpy as np
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import argparse
from os import path, listdir, makedirs


def create_model(model_name):
    # Layer Features
    if model_name == 'vgg16':
        layer_name = 'fc7/relu'
    elif model_name == 'resnet50':
        layer_name = 'flatten_1'
    else:
        raise Exception('Model name not recognized!')

    model = VGGFace(model=model_name)
    out = model.get_layer(layer_name).output

    return Model(model.input, out)


def extract_features(model_name, source, destination, weights=None):
    model = create_model(model_name)
    if weights is not None:
        model.load_weights(weights, by_name=True)

    if path.isfile(source):
        full_path = True
        source_list = np.sort(np.loadtxt(source, dtype=np.str))
    else:
        full_path = False
        source_list = listdir(source)

    if model_name == 'vgg16':
        version = 1
    else:
        version = 2

    for image_name in source_list:
        if not full_path:
            image_path = path.join(source, image_name)
        else:
            image_path = image_name
            image_name = path.split(image_name)[1]

        if not image_path.lower().endswith('.png') and not image_path.lower().endswith('.jpg') \
           and not image_path.lower().endswith('.bmp'):
            continue

        img = image.load_img(image_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = utils.preprocess_input(img, version=version)

        features = model.predict(img)

        dest_path = destination

        if full_path:
            sub_folder = path.basename(path.normpath(path.split(image_path)[0]))

            dest_path = path.join(destination, sub_folder)

            if not path.exists(dest_path):
                makedirs(dest_path)

        features_name = path.join(dest_path, image_name[:-3] + 'npy')

        np.save(features_name, features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features with CNN')
    parser.add_argument('--net', '-n', help='Net to run (vgg16 or resnet50.', default='resnet50')
    parser.add_argument('--source', '-s', help='Folder with images.')
    parser.add_argument('--dest', '-d', help='Folder to save the extractions.')
    parser.add_argument('--weights', '-w', help='Weight path for the network.', default=None)

    args = parser.parse_args()

    if not path.exists(args.dest):
        makedirs(args.dest)

    extract_features(args.net, args.source, args.dest)

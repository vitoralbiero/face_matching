from keras.engine import Model
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
import argparse
from os import path, listdir, makedirs


def create_model():
    model = ResNet50(include_top=False, input_shape=(224, 224, 3), weights=None, pooling='avg')

    return Model(model.input, model.output)


def preprocess(img):
    img = img[..., ::-1]
    img[..., 0] -= 91.4953
    img[..., 1] -= 103.8827
    img[..., 2] -= 131.0912

    return img


def extract_features(source, destination, weights=None):
    model = create_model()

    if weights is not None:
        print('Loading weights from {}'.format(weights))
        model.load_weights(weights, by_name=True)

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

        img = image.load_img(image_path, target_size=(224, 224))
        img = image.img_to_array(img)

        img = np.expand_dims(img, axis=0)

        if weights is not None:
            # img = img[..., ::-1]
            # img /= 255
            img = preprocess(img)

        if weights is None:
            img = preprocess_input(img)

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
    parser = argparse.ArgumentParser(description='Extract features with CNN')
    parser.add_argument('--source', '-s', help='Folder with images.')
    parser.add_argument('--dest', '-d', help='Folder to save the extractions.')
    parser.add_argument('--weights', '-w', help='Weight path for the network.', default=None)

    args = parser.parse_args()

    if not path.exists(args.dest):
        makedirs(args.dest)

    extract_features(args.source, args.dest, args.weights)

'''
Uses weights and models implementation' from
https://github.com/yule-li/CosFace
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import argparse
from os import path, listdir, makedirs
import cv2
import sys
sys.path.insert(0, '../../CosFace/networks')
sys.path.insert(0, '../../CosFace/lib')
import sphere_network as network
import utils


def extract_features(model, source, destination, image_height, image_width, prewhiten, fc_bn, feature_size):
    if path.isfile(source):
        full_path = True
        source_list = np.sort(np.loadtxt(source, dtype=np.str))
    else:
        full_path = False
        source_list = listdir(source)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            images_placeholder = tf.placeholder(tf.float32, shape=(None, image_height, image_width, 3), name='image')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            prelogits = network.infer(images_placeholder, feature_size)
            if fc_bn:
                prelogits = slim.batch_norm(prelogits, is_training=phase_train_placeholder,
                                            epsilon=1e-5, scale=True, scope='softmax_bn')

            embeddings = tf.identity(prelogits)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
            saver.restore(sess, model)

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

                images = utils.load_data([image_path], False, False, image_height,
                                         image_width, prewhiten, (image_height, image_width))
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                feats = sess.run(embeddings, feed_dict=feed_dict)

                feats = utils.l2_normalize(feats)

                np.save(features_name, feats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features with CNN')
    parser.add_argument('--source', '-s', help='Folder with images.')
    parser.add_argument('--dest', '-d', help='Folder to save the extractions.')
    parser.add_argument('--image_size', default='112,96', help='')
    parser.add_argument('--model', help='path to model.',
                        default='../../CosFace/models/model-20180309-083949.ckpt-60000')

    args = parser.parse_args()

    if not path.exists(args.dest):
        makedirs(args.dest)

    image_height = int(args.image_size.split(',')[0])
    image_width = int(args.image_size.split(',')[1])
    prewhiten = True
    fc_bn = False
    feature_size = 512

    extract_features(args.model, args.source, args.dest, image_height, image_width, prewhiten, fc_bn, feature_size)

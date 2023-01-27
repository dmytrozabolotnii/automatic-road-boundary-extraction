# Code written by Dmytro Zabolotnii, 2020/2021

"""
Evaluation of a single scene image using pre-trained Stage1 neural network without resizing to set square size
"""
import sys
import os
import os.path as osp
import argparse
import numpy as np
import tensorflow as tf
import cv2 as cv
from model_dilation_ver import resnext_fpn as resnext_fpn_dilated
from model_strides_ver import resnext_fpn as resnext_fpn_strides
from keras import optimizers, losses, callbacks
from typing import List, Tuple, Any


# Necessary hack for window tensorflow lib
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

X_SUFFIXES = ['_intensity', '_gradient']  # Suffixes used for the input data
Y_SUFFIXES = ['_distance']  # Suffixes used for the output data
IMAGE_SUFFIX = '.png'  # Suffixes used for the image
PIXEL_MAX_VALUE = 2 ** 8 - 1  # Is the images in 8 bit or 16 bit
CLASS_THRESHOLD = 0.1  # Relative pixel value threshold for classification
SAVE_FREQUENCY = 1  # Tensorflow log saving frequency
BATCH_SIZE = 4  # Base batch size (REQUIRES MORE THAN 8 GB OF GPU MEMORY with (512, 512) input size)
REDUCE_FACTOR = 2  # Still necessary due to memory restriction


def read_scene(scene_folder=osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'input', 'mini_set', 'scene-0061'),
               classification=False) -> (np.array, np.array, Tuple, Tuple):
    """
    Read a single scene images and prepare it for the neural network evaluation
    :param scene_folder: Path to the scene folder containing read scene
    :param classification: Whether dataset is used for classification or regression
    :return: Input images tensor, output images tensor, size used for evaluation, real size
    """

    scene_name = osp.basename(scene_folder)
    # Read image data and process it
    imgpath = osp.join(scene_folder, scene_name + X_SUFFIXES[0] + IMAGE_SUFFIX)
    image = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(float)
    # Target shape dividable by 32 to prevent padding problems
    real_shape = image.shape[:2]
    target_shape = ((image.shape[1] // REDUCE_FACTOR) - ((image.shape[1] // REDUCE_FACTOR) % 32),
                    (image.shape[0] // REDUCE_FACTOR) - ((image.shape[0] // REDUCE_FACTOR) % 32))
    image = cv.resize(image, target_shape, interpolation=cv.INTER_AREA)

    scene_size = image.shape[:2]
    data_x = np.zeros((1,) + tuple(scene_size) + tuple([len(X_SUFFIXES)]), dtype=np.float32)
    data_y = np.zeros((1,) + tuple(scene_size) + tuple([len(Y_SUFFIXES)]), dtype=np.float32)

    # Read image data and process it
    for j in range(len(X_SUFFIXES)):
        imgpath = osp.join(scene_folder, scene_name + X_SUFFIXES[0] + IMAGE_SUFFIX)
        image = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(float)
        image = cv.resize(image, target_shape, interpolation=cv.INTER_AREA)
        data_x[0, :, :, j] = image

    for j in range(len(Y_SUFFIXES)):
        imgpath = osp.join(scene_folder, scene_name + X_SUFFIXES[0] + IMAGE_SUFFIX)
        image = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(float)
        image = cv.resize(image, target_shape, interpolation=cv.INTER_AREA)
        # Normal processing
        # Make into two classes depending on threshold if we are using classification
        if classification:
            image[image < np.max(image) * CLASS_THRESHOLD] = 0
            image[image > 0] = 1
        data_y[0, :, :, j] = image

    # Normalize input data to mean 0, std 1
    for i in range(len(X_SUFFIXES)):
        data_x[:, :, :, i] = (data_x[:, :, :, i] - np.mean(data_x[:, :, :, i])) / \
                 (np.std(data_x[:, :, :, i]))
        print('Mean std values for input channel', i, ':', np.mean(data_x[:, :, :, i]), np.std(data_x[:, :, :, i]))

    # Print values
    for i in range(len(Y_SUFFIXES)):
        print('Max min values for labels channel', i, ':', np.max(data_y[:, :, :, i]), np.min(data_y[:, :, :, i]))
        print('Mean std values for labels channel', i, ':', np.mean(data_y[:, :, :, i]), np.std(data_y[:, :, :, i]))

    return data_x, data_y, scene_size, real_shape


def visualize_results(output, names, sizes, output_folder, classification=False):
    """
    Visualize results/ground truth of the training
    :param output: image to visualize
    :param names: names to save
    :param sizes: original size for reversing the resize
    :param output_folder: output folder name
    :param classification: whether to apply threshold for classification
    :return: None, saves the resized images to the specified names in the specified folder
    """
    # Visualize images one by one and save with given filename
    for i, name, size in zip(range(len(names) + 1), names, sizes):
        print('Saving ', name)
        if osp.join('stage1_patches', '') in name:
            name = name.replace(osp.join('stage1_patches', ''), '')
        distance = np.copy(output[i, :, :, 0:1])
        # Thresholding for classification
        if classification:
            distance[distance < np.max(distance) * CLASS_THRESHOLD] = 0
            distance[distance > 0] = 1
        distance = (PIXEL_MAX_VALUE) * (distance - np.min(distance)) / (np.max(distance) - np.min(distance))
        distance = cv.resize(distance, (size[1], size[0]), interpolation=cv.INTER_CUBIC)
        cv.imwrite(osp.join(output_folder, name + Y_SUFFIXES[0] + IMAGE_SUFFIX), distance.astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1 Full Size Eval Routine',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datafile_name', default='data.txt', type=str, help='Datafile name')
    parser.add_argument('--dataset_folder', default='mini_set', type=str, help='Folder name containing dataset to be trained on')
    parser.add_argument('--scene_name_index', default=0, type=int, help='Scene name index inside datafile')
    parser.add_argument('--model_mode', default=1, type=int, help='Mode for model, 0 - dilation model, '
                                                                  '1 - strides model')
    parser.add_argument('--classification', default=1, type=int, help='Use classification instead of regression')
    parser.add_argument('--reg_factor', default=0, type=float, help='Regularization factor')
    parser.add_argument('--batch_norm', default=0, type=int, help='Batch normalization flag')
    parser.add_argument('--val_data', default=1, type=int, help='Validation prefix flag')
    parser.add_argument('--out_dir', default='output', type=str, help='Output folder')
    parser.add_argument('--model_weights_name', default='', type=str, help='Name of the weights of model')

    args = parser.parse_args()
    DATAFILE_NAME = args.datafile_name
    DATASET_FOLDER = args.dataset_folder
    scene_name_index = int(args.scene_name_index)
    model_mode = bool(args.model_mode)
    classification = bool(args.classification)
    reg_factor = float(args.reg_factor)
    batch_norm = bool(args.batch_norm)
    val_data = bool(args.val_data)
    out_dir = args.out_dir
    model_weights_name = args.model_weights_name
    # Read data
    datafile_path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                             'input', DATASET_FOLDER, DATAFILE_NAME)
    with open(datafile_path) as f:
        scene_names = f.readlines()
    scene_names = [x.strip() for x in scene_names]

    scene_name = scene_names[scene_name_index]
    data_x, data_y, scene_size, scene_real_size = \
        read_scene(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                   'input', DATASET_FOLDER, scene_name), classification=classification)
    # Init model and compile it
    if model_mode == 1:
        model = resnext_fpn_strides(scene_size + (2,), depth=(2, 2, 2, 2), cardinality=1,
                                    reg_factor=reg_factor, batch_norm=batch_norm, classification=classification)
    else:
        model = resnext_fpn_dilated(scene_size + (2,), depth=(2, 2, 2, 2), cardinality=1,
                                    reg_factor=reg_factor, batch_norm=batch_norm, classification=classification)

    model_path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'models')
    if classification:
        model_path = osp.join(model_path, 'classification', out_dir)
        # Tensorboard hack
        model._get_distribution_strategy = lambda: None
        model.compile(optimizer=optimizers.Adam(learning_rate=0.00025), loss=losses.BinaryCrossentropy())
    else:
        model_path = osp.join(model_path, 'regression', out_dir)
        # Tensorboard hack
        model._get_distribution_strategy = lambda: None
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=losses.MeanSquaredError())
    if not osp.isdir(model_path):
        os.makedirs(model_path)
    # Evaluate model
    if not model_weights_name == '' and osp.isfile(osp.join(model_path, model_weights_name)):
        model.load_weights(osp.join(model_path, model_weights_name))
        print('Loaded model')
    else:
        sys.exit('Weights not available on specified path')
    output_path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', out_dir, 'evaluate_fullsize')

    print('Saving images to', output_path)
    # Create output folder
    if not output_path == '' and not osp.isdir(output_path):
        os.makedirs(output_path)
    output = model.predict(data_x)
    prefix = 'val_' if val_data else 'train_'
    visualize_results(output,
                      [prefix + scene_name],
                      [scene_real_size],
                      output_path, classification=classification)

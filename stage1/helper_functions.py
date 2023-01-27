# Code written by Dmytro Zabolotnii, 2020/2021

"""
Collection of helper function for reading and writing images for stage 1 neural network learning
"""


from os import path as osp

import cv2 as cv
import numpy as np
import itertools
import tensorflow as tf
from typing import List, Any

PIXEL_ORDER = 8  # Is the images in 8 bit or 16 bit
X_SUFFIXES = ['_intensity', '_gradient']  # Suffixes used for the input data
Y_SUFFIXES = ['_distance']  # Suffixes used for the output data
IMAGE_SUFFIX = '.png'  # Suffixes used for the image
CLASS_THRESHOLD = 0.1  # Relative pixel value threshold in case neural network is used for classification rather than regression


def augment_dataset(train_x: np.array, train_y: np.array, flips=True) -> (np.array, np.array):
    """
    Augment dataset by performing simplistic flips
    :param train_x: input data to augment
    :param train_y: output data to augment
    :param flips: trigger flips
    :return: augmented datasets with x4 size
    """
    if flips:
        # Do vertical horizontal and combination flip
        print('Doing flips augmentation')
        # Normal flips for input
        aug_x = np.concatenate((train_x, np.flip(train_x, 1), np.flip(train_x, 2), np.flip(np.flip(train_x, 2), 1)), axis=0)
        # Normal flips for output
        aug_y = np.concatenate((train_y, np.flip(train_y, 1), np.flip(train_y, 2), np.flip(np.flip(train_y, 2), 1)), axis=0)

        return aug_x, aug_y


def read_dataset(datapath=osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'input', 'mini_set', 'data.txt'),
                 input_shape=(512, 512), classification=False, validation_subset_size=0.1, augment=True) \
        -> (np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array):
    """
    Read dataset without reading patches, resize and pass it back to the training
    :param datapath: Path to the data file containing scene names used for training (see demo folder)
    :param input_shape: Resize target size
    :param classification: Whether dataset is used for classification or regression
    :param validation_subset_size: Validation subdataset part size
    :param augment: Whether to augment using flips
    :return: Train input array, Train output array, Val input array, Val output array,
     train scene names, validation scene names, train images original sizes, val images original sizes (for reverse resize)
    """
    # Read scene names from datafile
    with open(datapath) as f:
        scene_names = f.readlines()
    datapath = osp.dirname(datapath)
    scene_names = [x.strip() for x in scene_names]
    scene_sizes = []
    # Init data arrays
    dataset_size = len(scene_names)
    valset_size = int(dataset_size * validation_subset_size)
    data_x = np.zeros(tuple([dataset_size]) + input_shape + tuple([len(X_SUFFIXES)]), dtype=np.float32)
    data_y = np.zeros(tuple([dataset_size]) + input_shape + tuple([len(Y_SUFFIXES)]), dtype=np.float32)
    # Read every image data and process it
    for i in range(dataset_size):
        imgpath = osp.join(datapath, scene_names[i], scene_names[i] + X_SUFFIXES[0] + IMAGE_SUFFIX)
        image = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(float)
        scene_sizes.append(image.shape[:2])

        for j in range(len(X_SUFFIXES)):
            imgpath = osp.join(datapath, scene_names[i], scene_names[i] + X_SUFFIXES[j] + IMAGE_SUFFIX)
            image = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(float)
            image = cv.resize(image, input_shape, interpolation=cv.INTER_AREA)
            data_x[i, :, :, j] = image

        for j in range(len(Y_SUFFIXES)):
            imgpath = osp.join(datapath, scene_names[i], scene_names[i] + Y_SUFFIXES[j] + IMAGE_SUFFIX)
            image = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(float)
            # Normal processing
            image = cv.resize(image, input_shape, interpolation=cv.INTER_AREA)
            # Make into two classes depending on threshold if we are using classification
            if classification:
                image[image < np.max(image) * CLASS_THRESHOLD] = 0
                image[image > 0] = 1
            data_y[i, :, :, j] = image
    # Normalize input data to mean 0, std 1
    for i in range(len(X_SUFFIXES)):
        data_x[:, :, :, i] = (data_x[:, :, :, i] - np.mean(data_x[:, :, :, i])) / \
                 (np.std(data_x[:, :, :, i]))
        print('Mean std values for input channel', i, ':', np.mean(data_x[:, :, :, i]), np.std(data_x[:, :, :, i]))
    # Print values
    for i in range(len(Y_SUFFIXES)):
        print('Max min values for labels channel', i, ':', np.max(data_y[:, :, :, i]), np.min(data_y[:, :, :, i]))
        print('Mean std values for labels channel', i, ':', np.mean(data_y[:, :, :, i]), np.std(data_y[:, :, :, i]))
    # Divide into valset and trainset
    val_x = np.copy(data_x[:valset_size])
    val_y = np.copy(data_y[:valset_size])
    train_x = np.copy(data_x[valset_size:])
    train_y = np.copy(data_y[valset_size:])
    val_scene_names = np.array(scene_names)[:valset_size]
    train_scene_names = np.array(scene_names)[valset_size:]
    val_scene_sizes = np.array(scene_sizes)[:valset_size]
    train_scene_sizes = np.array(scene_sizes)[valset_size:]

    del data_x, data_y
    # Manual augmentation because we need to augment output feature maps too
    if augment:
        train_x, train_y = augment_dataset(train_x, train_y, flips=True)
        train_scene_names = np.concatenate((train_scene_names, train_scene_names,
                                            train_scene_names, train_scene_names), axis=0)

    return train_x, train_y, val_x, val_y, train_scene_names, val_scene_names, train_scene_sizes, val_scene_sizes


def read_patch_names(datapath: str, scene_name: str) -> List[str]:
    """
    Read names of the unspecified amount of the separate patches in the subfolder of the scene images
    :param datapath: Path to the general folder containing scene folders
    :param scene_name: Name of the specific scene whose patches names need to be read
    :return: List of patches names
    """
    patch_counter = 0
    scene_parts_names = []
    breakflag = True
    while breakflag:
        patch_name = scene_name + '_' + str(patch_counter)
        for j in range(len(X_SUFFIXES)):
            if not osp.isfile(osp.join(datapath, scene_name, 'stage1_patches', patch_name + X_SUFFIXES[j] + IMAGE_SUFFIX)):
                breakflag = False
        for j in range(len(Y_SUFFIXES)):
            if not osp.isfile(osp.join(datapath, scene_name, 'stage1_patches', patch_name + Y_SUFFIXES[j] + IMAGE_SUFFIX)):
                breakflag = False
        if breakflag:
            patch_counter += 1
            scene_parts_names.append(osp.join(scene_name, 'stage1_patches', patch_name))

    return scene_parts_names


def read_dataset_patches(datapath=osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'input', 'mini_set', 'data.txt'),
                 input_shape=(512, 512), classification=False, validation_subset_size=0.1, augment=True):
    """
    Read dataset and read patches, resize and pass it back to the training
    :param datapath: Path to the data file containing scene names used for training (see demo folder)
    :param input_shape: Resize target size
    :param classification: Whether dataset is used for classification or regression
    :param validation_subset_size: Validation subdataset part size
    :param augment: Whether to augment using flips
    :return: Train input array, Train output array, Val input array, Val output array,
     train scene names, validation scene names, train images original sizes, val images original sizes (for reverse resize)
    """
    # Read scene names from datafile
    with open(datapath) as f:
        scene_names = f.readlines()
    datapath = osp.dirname(datapath)
    scene_names = [x.strip() for x in scene_names]
    # Init data arrays sizes
    dataset_size = len(scene_names)
    valset_size = int(dataset_size * validation_subset_size)
    scene_parts_names_list = []
    # Read patch names and amount
    for i in range(dataset_size):
        # Read scene patches names and add the main one
        scene_parts_names = read_patch_names(datapath, scene_names[i])
        scene_parts_names.append(osp.join(scene_names[i], scene_names[i]))
        scene_parts_names_list.append(scene_parts_names)
    # Divide into train and val subset
    train_scene_names = list(itertools.chain.from_iterable(list(np.array(scene_parts_names_list, dtype=object)[valset_size:])))
    val_scene_names = list(itertools.chain.from_iterable(list(np.array(scene_parts_names_list, dtype=object)[:valset_size])))
    train_scene_sizes = []
    val_scene_sizes = []
    # Initiate arrays
    train_x = np.zeros(tuple([len(train_scene_names)]) + input_shape + tuple([len(X_SUFFIXES)]), dtype=np.float32)
    train_y = np.zeros(tuple([len(train_scene_names)]) + input_shape + tuple([len(Y_SUFFIXES)]), dtype=np.float32)
    val_x = np.zeros(tuple([len(val_scene_names)]) + input_shape + tuple([len(X_SUFFIXES)]), dtype=np.float32)
    val_y = np.zeros(tuple([len(val_scene_names)]) + input_shape + tuple([len(Y_SUFFIXES)]), dtype=np.float32)

    # Read every image data for train set and process it
    for count, scene_name in enumerate(train_scene_names):
        imgpath = osp.join(datapath, scene_name + X_SUFFIXES[0] + IMAGE_SUFFIX)
        image = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(float)
        train_scene_sizes.append(image.shape[:2])

        for j in range(len(X_SUFFIXES)):
            imgpath = osp.join(datapath, scene_name + X_SUFFIXES[j] + IMAGE_SUFFIX)
            image = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(float)
            image = cv.resize(image, input_shape, interpolation=cv.INTER_AREA)
            train_x[count, :, :, j] = image

        for j in range(len(Y_SUFFIXES)):
            imgpath = osp.join(datapath, scene_name + Y_SUFFIXES[j] + IMAGE_SUFFIX)
            image = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(float)
            # Normal processing
            image = cv.resize(image, input_shape, interpolation=cv.INTER_AREA)
            # Make into two classes depending on threshold if we are using classification
            if classification:
                image[image < np.max(image) * CLASS_THRESHOLD] = 0
                image[image > 0] = 1
            train_y[count, :, :, j] = image

    # Read every image data for val set and process it
    for count, scene_name in enumerate(val_scene_names):
        imgpath = osp.join(datapath, scene_name + X_SUFFIXES[j] + IMAGE_SUFFIX)
        image = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(float)
        val_scene_sizes.append(image.shape[:2])

        for j in range(len(X_SUFFIXES)):
            imgpath = osp.join(datapath, scene_name + X_SUFFIXES[j] + IMAGE_SUFFIX)
            image = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(float)
            image = cv.resize(image, input_shape, interpolation=cv.INTER_AREA)
            val_x[count, :, :, j] = image

        for j in range(len(Y_SUFFIXES)):
            imgpath = osp.join(datapath, scene_name + Y_SUFFIXES[j] + IMAGE_SUFFIX)
            image = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(float)
            # Normal processing
            image = cv.resize(image, input_shape, interpolation=cv.INTER_AREA)
            # Make into two classes depending on threshold if we are using classification
            if classification:
                image[image < np.max(image) * CLASS_THRESHOLD] = 0
                image[image > 0] = 1
            val_y[count, :, :, j] = image

    # Normalize input data to mean 0, std 1
    for i in range(len(X_SUFFIXES)):
        train_x[:, :, :, i] = (train_x[:, :, :, i] - np.mean(train_x[:, :, :, i])) / \
                 (np.std(train_x[:, :, :, i]))
        val_x[:, :, :, i] = (val_x[:, :, :, i] - np.mean(val_x[:, :, :, i])) / \
                 (np.std(val_x[:, :, :, i]))
        print('Mean std values for input channel', i, ':', np.mean(train_x[:, :, :, i]), np.std(train_x[:, :, :, i]))
    # Print values
    for i in range(len(Y_SUFFIXES)):
        print('Max min values for labels channel', i, ':', np.max(train_y[:, :, :, i]), np.min(train_y[:, :, :, i]))
        print('Mean std values for labels channel', i, ':', np.mean(train_y[:, :, :, i]), np.std(train_y[:, :, :, i]))
    # Manual augmentation because we need to augment output feature maps too
    if augment:
        train_x, train_y = augment_dataset(train_x, train_y, flips=True)
        train_scene_names = np.concatenate((train_scene_names, train_scene_names,
                                            train_scene_names, train_scene_names), axis=0)
        train_scene_sizes = np.concatenate((train_scene_sizes, train_scene_sizes,
                                            train_scene_sizes, train_scene_sizes), axis=0)
    val_scene_sizes = np.array(val_scene_sizes)
    train_scene_sizes = np.array(train_scene_sizes)

    return train_x, train_y, val_x, val_y, train_scene_names, val_scene_names, train_scene_sizes, val_scene_sizes


def visualize_results(output, names, sizes, output_folder, classification=False, file_writer=None) -> None:
    """
    Visualize results/ground truth of the training
    :param output: image to visualize
    :param names: names to save
    :param sizes: original size for reversing the resize
    :param output_folder: output folder name
    :param classification: whether to apply threshold for classification
    :param file_writer: tensorflowboard logging writer
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
        distance = (2 ** PIXEL_ORDER - 1) * (distance - np.min(distance)) / (np.max(distance) - np.min(distance))
        distance = cv.resize(distance, (size[1], size[0]), interpolation=cv.INTER_CUBIC)
        cv.imwrite(osp.join(output_folder, name + Y_SUFFIXES[0] + IMAGE_SUFFIX), distance.astype(np.uint16 if PIXEL_ORDER == 16 else np.uint8))
        if file_writer:
            with file_writer.as_default():
                tf.summary.image(name, distance[np.newaxis, :, :, np.newaxis], step=0)

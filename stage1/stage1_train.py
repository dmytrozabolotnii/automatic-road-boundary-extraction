# Code written by Dmytro Zabolotnii, 2020/2021

"""
Full dataset input reading/training routines/validation visualization for Stage 1 neural network for boundary detection
"""

import sys
import os
import os.path as osp
import time
import argparse
import numpy as np
import tensorflow as tf
from helper_functions import read_dataset, read_dataset_patches, visualize_results
from model_dilation_ver import resnext_fpn as resnext_fpn_dilated
from model_strides_ver import resnext_fpn as resnext_fpn_strides
from datetime import datetime
from keras import optimizers, losses, callbacks
from keras import backend as K


# Necessary hack for window tensorflow lib
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

INPUT_SHAPE = (512, 512)  # Base input shape (REQUIRES MORE THAN 8 GB OF GPU MEMORY)
PIXEL_ORDER = 8  # Is the images in 8 bit or 16 bit
CLASS_THRESHOLD = 0.1  # Relative pixel value threshold for classification and weighted losses
WEIGHTING = 0.04  # Weighting constant for WRMSE
SAVE_FREQUENCY = 1  # Tensorflow log saving frequency
BATCH_SIZE = 4  # Base batch size (REQUIRES MORE THAN 8 GB OF GPU MEMORY with (512, 512) input size)
VALIDATION_SUBSET = 0.25  # Default validation set size in relation to entire dataset


def root_mean_squared_error(y_true, y_pred):
    """
    RMSE error
    :param y_true: true values
    :param y_pred: predicted values
    :return: calculated loss
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def root_mean_squared_error_weighted(y_true, y_pred):
    """
    Unbalanced WRMSE error
    :param y_true: true values
    :param y_pred: predicted values
    :return: calculated loss
    """
    weights = tf.where(tf.math.greater_equal(y_true, tf.constant((2 ** PIXEL_ORDER - 1) * CLASS_THRESHOLD)), CLASS_THRESHOLD, (1 - CLASS_THRESHOLD))
    weights = tf.dtypes.cast(weights, tf.float32)

    return K.sqrt(K.sum(tf.math.multiply(K.square(y_pred - y_true), weights)) / K.sum(weights))


def root_mean_squared_error_weighted_balanced(y_true, y_pred):
    """
    Balanced WRMSE error (where weights add to 1)
    :param y_true: true values
    :param y_pred: predicted values
    :return: calculated loss
    """

    classes = tf.where(tf.math.greater_equal(y_true, tf.constant((2 ** PIXEL_ORDER - 1) * CLASS_THRESHOLD)), 1, 0)
    weights = tf.where(tf.math.greater_equal(y_true, tf.constant((2 ** PIXEL_ORDER - 1) * CLASS_THRESHOLD)), 1 / K.sum(classes), 1 / (K.sum(1 - classes)))
    weights = tf.dtypes.cast(weights, tf.float32)

    return K.sqrt(K.sum(tf.math.multiply(K.square(y_pred - y_true), weights)) / K.sum(weights))


def binary_crossentropy_weighted(y_true, y_pred):
    """
    Weighted binary crossentropy loss
    :param y_true: true values
    :param y_pred: predicted values
    :return: calculated loss
    """

    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * (WEIGHTING) + (1 - y_true) * K.log(1 - y_pred) * (1 - WEIGHTING))

    return K.mean(logloss, axis=-1)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Stage 1 Training Routine',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--datafile_name', default='data.txt', type=str, help='Datafile name containing names of training and validation scenes')
    parser.add_argument('--dataset_folder', default='mini_set', type=str, help='Folder name containing dataset to be trained on')
    parser.add_argument('--model_mode', default=1, type=int, help='Mode for model, 0 - dilation model, '
                                                                  '1 - strides model')
    parser.add_argument('--classification', default=1, type=int, help='Use classification instead of regression')
    parser.add_argument('--patches', default=0, type=int, help='Use patch augmented dataset')
    parser.add_argument('--reg_factor', default=0, type=float, help='Regularization factor')
    parser.add_argument('--batch_norm', default=0, type=int, help='Batch normalization flag')
    parser.add_argument('--out_dir', default='output', type=str, help='Output folder for the example visualizations')
    parser.add_argument('--training_mode', default=1, type=int, help='Mode for training, 1 - normal training, '
                                                                     '0 just run fast evaluation')
    parser.add_argument('--model_weights_name', default='', type=str, help='Name of the weights of model for run evaluation')

    args = parser.parse_args()
    EPOCHS = args.epochs
    DATAFILE_NAME = args.datafile_name
    DATASET_FOLDER = args.dataset_folder
    model_mode = bool(args.model_mode)
    classification = bool(args.classification)
    patches = bool(args.patches)
    reg_factor = float(args.reg_factor)
    batch_norm = bool(args.batch_norm)
    out_dir = args.out_dir
    training_mode = args.training_mode
    model_weights_name = args.model_weights_name
    # Read data
    if not patches:
        train_x, train_y, val_x, val_y, train_scene_names, val_scene_names, train_scene_sizes, val_scene_sizes = \
            read_dataset(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                         'input', DATASET_FOLDER, DATAFILE_NAME), input_shape=INPUT_SHAPE,
                         validation_subset_size=VALIDATION_SUBSET, augment=True, classification=classification)
    else:
        train_x, train_y, val_x, val_y, train_scene_names, val_scene_names, train_scene_sizes, val_scene_sizes = \
            read_dataset_patches(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                                 'input', DATASET_FOLDER, DATAFILE_NAME), input_shape=INPUT_SHAPE,
                                 validation_subset_size=VALIDATION_SUBSET, augment=False, classification=classification)
    # Init model and compile it
    if model_mode == 1:
        model = resnext_fpn_strides(INPUT_SHAPE + (2,), depth=(2, 2, 2, 2), cardinality=1,
                                    reg_factor=reg_factor, batch_norm=batch_norm, classification=classification)
    else:
        model = resnext_fpn_dilated(INPUT_SHAPE + (2,), depth=(2, 2, 2, 2), cardinality=1,
                                    reg_factor=reg_factor, batch_norm=batch_norm, classification=classification)

    model_path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'models')
    if classification:
        model_path = osp.join(model_path, 'classification', out_dir)
        # Tensorboard hack
        model._get_distribution_strategy = lambda: None
        model.compile(optimizer=optimizers.Adam(learning_rate=0.00025), loss=binary_crossentropy_weighted)
    else:
        model_path = osp.join(model_path, 'regression', out_dir)
        # Tensorboard hack
        model._get_distribution_strategy = lambda: None
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=root_mean_squared_error_weighted)
    if not osp.isdir(model_path):
        os.makedirs(model_path)
    # Write logs folder
    if training_mode == 1:
        logs_path = os.path.join(model_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-logs')
    elif training_mode == 0:
        logs_path = os.path.join(model_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-logs-evaluate')

    # Save model on every Xth epoch with loss
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=osp.join(model_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '_' + str(DATAFILE_NAME[:-4]) + '_{epoch:02d}_{val_loss:.2f}.hdf5'),
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        period=SAVE_FREQUENCY)
    # Add tensorboard callback and writer
    tensorboard = callbacks.TensorBoard(log_dir=logs_path,
                                        histogram_freq=5,
                                        batch_size=BATCH_SIZE,
                                        write_graph=True,
                                        write_grads=False,
                                        write_images=False)
    file_writer = tf.summary.create_file_writer(logs_path)
    # Train model
    if training_mode == 1:
        print('Saving models to', model_path)
        history = model.fit(train_x, train_y,
                            batch_size=BATCH_SIZE, epochs=EPOCHS,
                            validation_data=(val_x, val_y),
                            callbacks=[model_checkpoint_callback, tensorboard],
                            shuffle=True,
                            verbose=2)
        # Save model for future use
        model.save(osp.join(model_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '_' + str(DATAFILE_NAME[:-4]) + '_' + '{:02d}'.format(EPOCHS) + '_' +
                            '{:.2f}'.format(history.history['val_loss'][-1]) + '.hdf5'))
    elif training_mode == 0:
        if not model_weights_name == '' and osp.isfile(osp.join(model_path, model_weights_name)):
            model.load_weights(osp.join(model_path, model_weights_name))
            print('Loaded model')
        else:
            sys.exit('Weights not available on specified path')
    # mode 2
    # Visualize some training and validation images
    if training_mode == 1:
        output_path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', out_dir)
    elif training_mode == 0:
        output_path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', out_dir, 'evaluate')
    print('Saving images to', output_path)
    # Create output folder
    if not output_path == '' and not osp.isdir(output_path):
        os.makedirs(output_path)
    if training_mode == 1:
        # Visualize 4 examples from validation and training subdatasets
        viz_start = 0
        viz_end = 3
        test_output = model.predict(train_x[viz_start:viz_end])
        visualize_results(test_output, ['train_' + osp.basename(train_scene_names[i]) for i in range(viz_start, viz_end)],
                          train_scene_sizes[viz_start:viz_end],
                          output_path, classification=classification, file_writer=file_writer)
        visualize_results(train_y[viz_start:viz_end, :, :, :],
                          ['train_gt_' + osp.basename(train_scene_names[i]) for i in range(viz_start, viz_end)],
                          train_scene_sizes[viz_start:viz_end],
                          output_path, classification=classification, file_writer=file_writer)
        test_output = model.predict(val_x[viz_start:viz_end])
        visualize_results(test_output, ['val_' + osp.basename(val_scene_names[i]) for i in range(viz_start, viz_end)],
                          val_scene_sizes[viz_start:viz_end],
                          output_path, classification=classification, file_writer=file_writer)
        visualize_results(val_y[viz_start:viz_end, :, :, :],
                          ['val_gt_' + osp.basename(val_scene_names[i]) for i in range(viz_start, viz_end)],
                          val_scene_sizes[viz_start:viz_end],
                          output_path, classification=classification, file_writer=file_writer)
    elif training_mode == 0:
        # Visualize entire training/validation datasets, evaluating 4 at a time due to memory restrictions
        for i in range(len(train_scene_sizes) // 4 + 1):
            viz_start = min(i * 4, len(train_scene_sizes) - 1)
            viz_end = min((i + 1) * 4, len(train_scene_sizes))
            test_output = model.predict(train_x[viz_start:viz_end])
            visualize_results(test_output,
                              ['train_' + osp.basename(train_scene_names[i]) for i in range(viz_start, viz_end)],
                              train_scene_sizes[viz_start:viz_end],
                              output_path, classification=classification, file_writer=None)
        for i in range(len(val_scene_sizes) // 4 + 1):
            viz_start = min(i * 4, len(val_scene_sizes) - 1)
            viz_end = min((i + 1) * 4, len(val_scene_sizes))

            test_output = model.predict(val_x[viz_start:viz_end])
            visualize_results(test_output,
                              ['val_' + osp.basename(val_scene_names[i]) for i in range(viz_start, viz_end)],
                              val_scene_sizes[viz_start:viz_end],
                              output_path, classification=classification, file_writer=None)

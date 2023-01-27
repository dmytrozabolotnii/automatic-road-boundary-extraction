# Code written by Dmytro Zabolotnii, 2020/2021

"""
Processes generated road boundary images from Stage 1 with morphological operations.
Combines resultant Stage 2 images with ground truth images, and input intensity/gradient images into demo images
"""

import os
from os import path as osp
import shutil
import cv2 as cv
import numpy as np
import argparse
from tqdm import tqdm
from scipy import ndimage
from skimage.morphology import skeletonize, closing, dilation, disk, erosion, opening
from fil_finder import FilFinder2D
import astropy.units as u
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18, 6)

PIXEL_ORDER = 8  # Is the images in 8 bit or 16 bit
X_SUFFIXES = ['_intensity', '_gradient']  # Suffixes used for the input data
Y_SUFFIXES = ['_distance']  # Suffixes used for the output data
Y_PREFIXES = ['train_', 'val_']   # Prefixes used for the output data
IMAGE_SUFFIX = '.png'  # Suffixes used for the image
CLASS_THRESHOLD = 0.045
CONVOLUTION_SIZE = 10
CONVOLUTION_SIZE_COMPARISON = 20


def calculate_confusion_matrix(gen_, gt_, threshold=CLASS_THRESHOLD):
    gen = np.copy(gen_)
    gt = np.copy(gt_)
    a = disk(CONVOLUTION_SIZE_COMPARISON)
    # Threshold values
    if threshold is not 0:
        gen[gen < np.max(gen) * threshold] = 0
        gen[gen > 0] = 1
        gt[gt < np.max(gt) * threshold] = 0
        gt[gt > 0] = 1
    # Convolve generated boundaries for degree of accuracy adjustment for TP and FN
    gen_convol = ndimage.convolve(gen, a, mode='constant', cval=0.0)
    TP = np.sum(np.logical_and(gt, gen_convol))
    FN = np.sum(np.logical_and(gt, np.logical_not(gen_convol)))
    # Convolve ground truth boundaries for degree of accuracy adjustment for FP and TN
    gt_convol = ndimage.convolve(gt, a, mode='constant', cval=0.0)
    FP = np.sum(np.logical_and(gen, np.logical_not(gt_convol)))
    TN = np.sum(np.logical_and(np.logical_not(gen), np.logical_not(gt_convol)))
    # Return normalized ratios
    total = TP + FP + FN + TN

    return np.array([TP / total, FP / total, FN / total, TN / total])


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Stage 2 morphological process',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gen_dataset_folder', type=str,
                        help='Folder name containing image output of Stage 1 neural network', required=True)
    parser.add_argument('--datafile_name', default='data.txt', type=str,
                        help='Datafile name containing names of scenes that needs to be processed using Stage 2')
    parser.add_argument('--gt_dataset_folder', default='mini_set', type=str,
                        help='Folder name containing ground truth data generated from Nuscenes')
    parser.add_argument('--out_dir', default='output', type=str,
                        help='Folder name for the Stage 2 visualization result')
    parser.add_argument('--adaptive_thresholding', default=1, type=int, help='Use adaptive gaussian thresholding instead of set')

    args = parser.parse_args()
    DATAFILE_NAME = args.datafile_name
    GEN_DATASET_FOLDER = args.gen_dataset_folder
    GT_DATASET_FOLDER = args.gt_dataset_folder
    out_dir = args.out_dir
    adaptive_thresholding = bool(args.adaptive_thresholding)

    # Read scene names from datafile
    with open(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                         'output', GEN_DATASET_FOLDER, DATAFILE_NAME)) as f:
        scene_names = f.readlines()
    gt_datapath = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                         'input', GT_DATASET_FOLDER)
    gen_datapath = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                         'output', GEN_DATASET_FOLDER)
    out_datapath = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                         'demo', out_dir)
    if not out_datapath == '' and not osp.isdir(out_datapath):
        os.makedirs(out_datapath)

    scene_names = [x.strip() for x in scene_names]
    # Init data arrays
    # Init confusion matrixes for every threshold if not adaptive
    if not adaptive_thresholding:
        thresholds = [0.02 * i for i in range(1, 6)]
    else:
        thresholds = [1]

    confusion_matrixes = np.zeros((len(thresholds), len(scene_names), 4))

    for i in tqdm(range(len(scene_names))):
        imgpath = osp.join(gt_datapath, scene_names[i], scene_names[i] + X_SUFFIXES[0] + IMAGE_SUFFIX)
        if not osp.isfile(imgpath):
            print('Scene ' + scene_names[i] + ' is missing')
            continue
        image_base = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(np.uint16 if PIXEL_ORDER == 16 else np.uint8)
        imgpath = osp.join(gt_datapath, scene_names[i], scene_names[i] + X_SUFFIXES[1] + IMAGE_SUFFIX)
        image_base_2 = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(np.uint16 if PIXEL_ORDER == 16 else np.uint8)
        imgpath = osp.join(gt_datapath, scene_names[i], scene_names[i] + Y_SUFFIXES[0] + IMAGE_SUFFIX)
        image_gt = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(np.uint16 if PIXEL_ORDER == 16 else np.uint8)
        if osp.isfile(osp.join(gen_datapath, Y_PREFIXES[0] + scene_names[i] + Y_SUFFIXES[0] + IMAGE_SUFFIX)):
            imgpath = osp.join(gen_datapath, Y_PREFIXES[0] + scene_names[i] + Y_SUFFIXES[0] + IMAGE_SUFFIX)
            outpath = osp.join(out_datapath, Y_PREFIXES[0] + scene_names[i] + '_demo' + IMAGE_SUFFIX)
        else:
            imgpath = osp.join(gen_datapath, Y_PREFIXES[1] + scene_names[i] + Y_SUFFIXES[0] + IMAGE_SUFFIX)
            outpath = osp.join(out_datapath, Y_PREFIXES[1] + scene_names[i] + '_demo' + IMAGE_SUFFIX)
        image_gen = cv.imread(imgpath, cv.IMREAD_UNCHANGED).astype(np.uint16 if PIXEL_ORDER == 16 else np.uint8)
        # Create inverse intensity mask and convolute it to cut edge predictions
        selem = disk(CONVOLUTION_SIZE)
        selem2x = disk(CONVOLUTION_SIZE * 2)
        selem4x = disk(CONVOLUTION_SIZE * 4)
        mask = image_base == 0
        mask = opening(mask, selem)
        mask = ndimage.convolve(mask, selem2x, mode='constant', cval=0.0)
        mask = mask == 0
        # Calculate confusion matrix for scene with adaptive threshold and closing
        image_gen = cv.adaptiveThreshold(image_gen, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, CONVOLUTION_SIZE + 1, -2)
        image_gen = dilation(image_gen, selem2x)
        image_gen = skeletonize(image_gen > 0) * (2 ** PIXEL_ORDER - 1)
        fil = FilFinder2D(image_gen, mask=image_gen)
        fil.create_mask(use_existing_mask=True)
        fil.medskel(verbose=False)
        fil.analyze_skeletons(branch_thresh=40 * u.pix, skel_thresh=20 * u.pix, prune_criteria='length')
        image_gen = fil.skeleton_longpath * (2 ** PIXEL_ORDER - 1)
        image_gen = image_gen * mask
        # Filter out very small connected regions
        num_labels, labels_im = cv.connectedComponents(image_gen.astype(np.uint8), connectivity=8)
        for k in range(1, np.max(labels_im) + 1):
            if np.sum(labels_im == k) <= CONVOLUTION_SIZE * 4:
                labels_im[labels_im == k] = 0
        image_gen_endpoint_temp = np.copy(image_gen * (labels_im > 0))
        # Road completion algorithm implementation
        # Calculate endpoints
        num_labels, labels_im = cv.connectedComponents(image_gen_endpoint_temp.astype(np.uint8), connectivity=8)
        a = np.ones((3, 3))
        convolved = ndimage.convolve(labels_im, a, mode='constant', cval=0.0)
        endpoints = [np.reshape(np.argwhere((convolved * (labels_im == i)) == 2 * i), (-1, 1, 2)).astype(np.int32) for i in range(1, np.max(labels_im) + 1)]
        # Remove circular shapes without specifically 2 endpoints
        for k, endpoint in enumerate(endpoints):
            if endpoint.size != 4:
                labels_im[labels_im == (k + 1)] = 0
        image_gen_endpoint_temp = image_gen_endpoint_temp * (labels_im > 0)
        # Recalculate endpoints
        num_labels, labels_im = cv.connectedComponents(image_gen_endpoint_temp.astype(np.uint8), connectivity=8)
        a = np.ones((3, 3))
        convolved = ndimage.convolve(labels_im, a, mode='constant', cval=0.0)
        endpoints = [np.reshape(np.argwhere((convolved * (labels_im == i)) == 2 * i), (-1, 1, 2)).astype(np.int32) for i in range(1, np.max(labels_im) + 1)]
        # Calculate extended enpoints
        extended_endpoins_map_1 = np.zeros_like(labels_im)
        extended_endpoins_map_2 = np.zeros_like(labels_im)
        for k, endpoint in enumerate(endpoints):
            # Find extended endpoints for one endpoint of line
            extended_endpoins_map_1[endpoint[0, 0, 0], endpoint[0, 0, 1]] = 2 * (k + 1)
            range_x0 = max(endpoint[0, 0, 0] - 4 * CONVOLUTION_SIZE, 0)
            selem_range_x0 = -1 * min(endpoint[0, 0, 0] - 4 * CONVOLUTION_SIZE, 0)
            range_x1 = min(endpoint[0, 0, 0] + 4 * CONVOLUTION_SIZE + 1, extended_endpoins_map_1.shape[0])
            range_y0 = max(endpoint[0, 0, 1] - 4 * CONVOLUTION_SIZE, 0)
            selem_range_y0 = -1 * min(endpoint[0, 0, 1] - 4 * CONVOLUTION_SIZE, 0)
            range_y1 = min(endpoint[0, 0, 1] + 4 * CONVOLUTION_SIZE + 1, extended_endpoins_map_1.shape[1])
            # Search neighbourhood and isolate the endpoint connected points
            local_area = np.bitwise_and(selem4x[selem_range_x0:selem_range_x0+range_x1-range_x0, selem_range_y0:selem_range_y0+range_y1-range_y0],
                                        (k + 1) == labels_im[range_x0:range_x1, range_y0:range_y1])
            _, labels_im_local = cv.connectedComponents(local_area.astype(np.uint8), connectivity=8)
            labels_im_local = labels_im_local == labels_im_local[4 * CONVOLUTION_SIZE - selem_range_x0, 4 * CONVOLUTION_SIZE - selem_range_y0]
            extended_endpoins_map_1[range_x0:range_x1, range_y0:range_y1] = 2 * (k + 1) * labels_im_local

            # Find extended endpoints for another endpoint of line
            extended_endpoins_map_2[endpoint[1, 0, 0], endpoint[1, 0, 1]] = 2 * (k + 1) + 1
            range_x0 = max(endpoint[1, 0, 0] - 4 * CONVOLUTION_SIZE, 0)
            selem_range_x0 = -1 * min(endpoint[1, 0, 0] - 4 * CONVOLUTION_SIZE, 0)
            range_x1 = min(endpoint[1, 0, 0] + 4 * CONVOLUTION_SIZE + 1, extended_endpoins_map_2.shape[0])
            range_y0 = max(endpoint[1, 0, 1] - 4 * CONVOLUTION_SIZE, 0)
            selem_range_y0 = -1 * min(endpoint[1, 0, 1] - 4 * CONVOLUTION_SIZE, 0)
            range_y1 = min(endpoint[1, 0, 1] + 4 * CONVOLUTION_SIZE + 1, extended_endpoins_map_2.shape[1])
            # Search neighbourhood and isolate the endpoint connected points
            local_area = np.bitwise_and(selem4x[selem_range_x0:selem_range_x0+range_x1-range_x0, selem_range_y0:selem_range_y0+range_y1-range_y0],
                                        (k + 1) == labels_im[range_x0:range_x1, range_y0:range_y1])
            _, labels_im_local = cv.connectedComponents(local_area.astype(np.uint8), connectivity=8)
            labels_im_local = labels_im_local == labels_im_local[4 * CONVOLUTION_SIZE - selem_range_x0, 4 * CONVOLUTION_SIZE - selem_range_y0]
            extended_endpoins_map_2[range_x0:range_x1, range_y0:range_y1] = (2 * (k + 1) + 1) * labels_im_local

        extended_endpoints = [[np.reshape(np.argwhere(extended_endpoins_map_1 == 2 * i), (-1, 1, 2)).astype(np.int32),
                               np.reshape(np.argwhere(extended_endpoins_map_2 == 2 * i + 1), (-1, 1, 2)).astype(np.int32)] for i in range(1, np.max(labels_im + 1))]
        # Calculate contours, join, and sort by contour length
        contours = [np.reshape(np.argwhere(labels_im == i), (-1, 1, 2)).astype(np.int32) for i in range(1, np.max(labels_im) + 1)]
        joined_contours = sorted(zip(contours, endpoints, extended_endpoints), key=lambda join: join[0].shape[0], reverse=True)

        for j in range(len(joined_contours)):
            # For every contour starting from biggest, calculate minimum distances to all others
            distances = []
            routes = []
            destination = []
            for k in range(len(joined_contours)):
                if j != k:
                    local_distances = [
                                       np.linalg.norm(joined_contours[j][1][0] - joined_contours[k][1][0]),
                                       np.linalg.norm(joined_contours[j][1][0] - joined_contours[k][1][1]),
                                       np.linalg.norm(joined_contours[j][1][1] - joined_contours[k][1][0]),
                                       np.linalg.norm(joined_contours[j][1][1] - joined_contours[k][1][1])]
                    distances.append(np.min(local_distances))
                    routes.append(np.argmin(local_distances))
                    destination.append(k)
            joined_distances = sorted(zip(distances, routes, destination), key=lambda distance: distance[0], reverse=True)
            # Start joining from smallest distance
            for joined_distance in joined_distances:
                if joined_contours[j][2][joined_distance[1] // 2].size != 0 and \
                   joined_contours[joined_distance[2]][2][joined_distance[1] % 2].size != 0:
                    (_, _), (width, height), _ = cv.minAreaRect(np.concatenate((
                        joined_contours[j][2][joined_distance[1] // 2][:, :, ::-1],
                        joined_contours[joined_distance[2]][2][joined_distance[1] % 2][:, :, ::-1]
                    )))
                    if min(width, height) < 6 and max(width, height) < 500:
                        cv.line(image_base_2, tuple(joined_contours[j][1][joined_distance[1] // 2][:, ::-1][0]),
                                tuple(joined_contours[joined_distance[2]][1][joined_distance[1] % 2][:, ::-1][0]), (255, 255, 255), 1)
                        cv.line(image_gen, tuple(joined_contours[j][1][joined_distance[1] // 2][:, ::-1][0]),
                                tuple(joined_contours[joined_distance[2]][1][joined_distance[1] % 2][:, ::-1][0]), (255, 255, 255), 1)

                        joined_contours[j][2][joined_distance[1] // 2] = np.array([])
                        joined_contours[joined_distance[2]][2][joined_distance[1] % 2] = np.array([])
                        break
        # Road completion end
        # Filter out new skeleton loops
        fil = FilFinder2D(image_gen, mask=image_gen)
        fil.create_mask(use_existing_mask=True)
        fil.medskel(verbose=False)
        fil.analyze_skeletons(branch_thresh=40 * u.pix, skel_thresh=20 * u.pix, prune_criteria='length')
        image_gen = fil.skeleton_longpath * (2 ** PIXEL_ORDER - 1)
        # Filter out medium connected regions
        num_labels, labels_im = cv.connectedComponents(image_gen.astype(np.uint8), connectivity=8)
        for k in range(1, np.max(labels_im) + 1):
            if np.sum(labels_im == k) <= CONVOLUTION_SIZE * 8:
                labels_im[labels_im == k] = 0
        image_gen = image_gen * (labels_im > 0)

        # Calculate confusion matrixes for the thresholds
        ret, image_gt = cv.threshold(image_gt, 127, 255, cv.THRESH_BINARY)
        image_gt = skeletonize(image_gt > 0) * (2 ** PIXEL_ORDER - 1)
        if adaptive_thresholding:
            for j in range(len(thresholds)):
                confusion_matrixes[j, i] = calculate_confusion_matrix(image_gen, image_gt, 0)
        else:
            for j in range(len(thresholds)):
                confusion_matrixes[j, i] = calculate_confusion_matrix(image_gen, image_gt, thresholds[j])

        # Overlap ground truth as red channel, generated boundaries as green channel,
        # intensity data as grayscale, gradient data as blue channel
        image_out = cv.cvtColor(image_base, cv.COLOR_GRAY2BGR)
        image_out[:, :, 0] = np.maximum(image_out[:, :, 0], image_base_2[:, :])
        image_out[:, :, 1] = np.maximum(image_out[:, :, 1], image_gen)
        image_out[:, :, 2] = np.maximum(image_out[:, :, 2], image_gt[:, :])
        cv.imwrite(outpath, image_out.astype(np.uint16 if PIXEL_ORDER == 16 else np.uint8))
    # Output accuracy, precision, recall and f1 metrics
    avgacc, avgpre, avgrec, avgf1 = [], [], [], []
    for j in range(len(thresholds)):
        accuracy_array = np.array([(i[0] + i[3]) / np.sum(i) for i in confusion_matrixes[j]])
        precision_array = np.array([(i[0]) / (i[0] + i[1]) for i in confusion_matrixes[j]])
        recall_array = np.array([i[0] / (i[0] + i[2]) for i in confusion_matrixes[j]])
        f1_array = 2 * precision_array * recall_array / (precision_array + recall_array)
        print('Threshold:', thresholds[j])
        avgacc.append(np.mean(accuracy_array))
        print('Average accuracy:', np.mean(accuracy_array))
        avgpre.append(np.mean(precision_array))
        print('Average precision:', np.mean(precision_array))
        print('90th percentile', np.percentile(precision_array, 90))
        avgrec.append(np.mean(recall_array))
        print('Average recall:', np.mean(recall_array))
        print('90th percentile', np.percentile(recall_array, 90))
        avgf1.append(np.mean(f1_array))
        print('Average f1 score:', np.mean(f1_array))
        print('Average confusion matrix:')
        print(np.sum(confusion_matrixes[j], axis=0) / np.sum(confusion_matrixes[j]))


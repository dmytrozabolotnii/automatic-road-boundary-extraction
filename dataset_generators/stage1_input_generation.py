# Credits:
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/map_expansion/map_api.py#L759
# nuScenes dev-kit.
# Code written by Holger Caesar, 2018. Dmytro Zabolotnii, 2020/2021

"""
Generates input and ground truth images for stage 1 of custom hd map road boundaries detection neural network
from nuscenes data set pointclouds
Is unfortunately pretty slow during the generation of ground truth images, with ground truth generation of 1 scene
taking up to 15-20 min
"""

import argparse
import os
import os.path as osp
from typing import List, Any

import numpy as np
import cv2 as cv
from pyquaternion import Quaternion
from shapely.geometry import Point, box, LineString, MultiLineString
from shapely.ops import nearest_points
from tqdm import tqdm
import functools
from scipy.spatial import cKDTree

from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import LidarPointCloud, Box

PIXEL_ORDER = 8  # Is the images in 8 bit or 16 bit
STANDARD_RESOLUTION = 0.04  # The size of one pixel cell in meters
MIN_TOUCH_DISTANCE = 1e-3  # Touch distance between partial road boundaries
TQDM_DISABLE = True  # Disable tqdm for cluster processing
BOX_MAX = 30  # Minimum size of the outer box


def export_scene_pointcloud(nusc: NuScenes,
                            scene_token: str,
                            channel: str = 'LIDAR_TOP',
                            min_dist: float = 2.0,
                            max_dist: float = 200.0,
                            z_filter: bool = True) -> np.array:
    """
    Export fused point clouds of a scene to numpy array containing all points
    :param nusc: NuScenes instance.
    :param scene_token: Unique identifier of scene to render.
    :param channel: Channel to render.
    :param min_dist: Minimum distance to ego vehicle below which points are dropped.
    :param max_dist: Maximum distance to ego vehicle above which points are dropped.
    :param z_filter: Do we filter non-ground points depending on height from sensor

    Return array containing all points in cloud and box coordinates that contain all points in 2d
    """
    # Check inputs.
    valid_channels = ['LIDAR_TOP', 'RADAR_FRONT', 'RADAR_FRONT_RIGHT', 'RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT',
                      'RADAR_BACK_RIGHT']
    assert channel in valid_channels, 'Input channel {} not valid.'.format(channel)

    # Get records from DB.
    scene_rec = nusc.get('scene', scene_token)
    start_sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
    sd_rec = nusc.get('sample_data', start_sample_rec['data'][channel])

    # Make list of frames
    cur_sd_rec = sd_rec
    sd_tokens = []
    while cur_sd_rec['next'] != '':
        cur_sd_rec = nusc.get('sample_data', cur_sd_rec['next'])
        sd_tokens.append(cur_sd_rec['token'])

    # Initiate pointcloud box
    poincloud_box = np.array([np.inf, np.inf, -1 * np.inf, -1 * np.inf])

    # Generate combined pointcloud
    pc_total = None
    poserecord_history = []
    for sd_token in tqdm(sd_tokens, disable=TQDM_DISABLE):
        lidar_token = sd_rec['token']
        lidar_rec = nusc.get('sample_data', lidar_token)
        pc = LidarPointCloud.from_file(osp.join(nusc.dataroot, lidar_rec['filename']))

        # Points live in their own reference frame. So they need to be transformed via global to the image plane.
        # First step: transform the point cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', lidar_rec['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Optional Filter by distance to remove the ego vehicle.
        dists_origin = np.sqrt(np.sum(pc.points[:3, :] ** 2, axis=0))
        keep = np.logical_and(min_dist <= dists_origin, dists_origin <= max_dist)
        pc.points = pc.points[:, keep]
        # Optional Filter by z-axis to remove trees and building
        if z_filter:
            pc.points = pc.points[:, pc.points[2, :] < 0.5]

        # Second step: transform to the global frame.
        poserecord = nusc.get('ego_pose', lidar_rec['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))
        poserecord_history.append(poserecord['translation'])
        # Filter to remove points in annotations bboxes
        boxes = nusc.get_boxes(sd_token)
        for box in boxes:
            keep = outside_test(box, pc.points[:3, :].T)
            pc.points = pc.points[:, keep]

        if pc_total is None:
            pc_total = np.copy(pc.points)
        else:
            pc_total = np.concatenate((pc_total, pc.points), axis=1)

        if not sd_rec['next'] == "":
            sd_rec = nusc.get('sample_data', sd_rec['next'])

    poserecord_history = np.array(poserecord_history)
    poincloud_box[0] = min(poincloud_box[0], np.min(poserecord_history[:, 0]) - BOX_MAX)
    poincloud_box[1] = min(poincloud_box[1], np.min(poserecord_history[:, 1]) - BOX_MAX)
    poincloud_box[2] = max(poincloud_box[2], np.max(poserecord_history[:, 0]) + BOX_MAX)
    poincloud_box[3] = max(poincloud_box[3], np.max(poserecord_history[:, 1]) + BOX_MAX)

    return pc_total.T, poincloud_box, poserecord_history


def outside_test(box: Box,
                 points: np.array) -> np.array:
    """
    Find the points that are outside custom cube box
    :param box  =  Box custom object of nuscenes
    :param points = numpy array of points with shape (N, 3).

    Returns the indices of the points array which are outside the cube3d
    """
    corners = box.corners().T

    origin = corners[7]
    xpoint = corners[6]
    ypoint = corners[2]
    zpoint = corners[4]

    dir1 = (xpoint - origin)
    dir1 = dir1 / np.sum(dir1 ** 2) ** 0.5

    dir2 = (ypoint - origin)
    dir2 = dir2 / np.sum(dir2 ** 2) ** 0.5

    dir3 = (zpoint - origin)
    dir3 = dir3 / np.sum(dir3 ** 2) ** 0.5

    dir_vec = points - box.center

    res = np.logical_or((np.absolute(np.dot(dir_vec, dir1)) * 2) > box.wlh[0],
                        (np.absolute(np.dot(dir_vec, dir2)) * 2) > box.wlh[1])
    res = np.logical_or(res, (np.absolute(np.dot(dir_vec, dir3)) * 2) > box.wlh[2])

    return res


def generate_random_patch(pointcloud_box: np.array,
                          min_size=(256, 256),
                          resolution=STANDARD_RESOLUTION) -> np.array:
    """
    Generate random patch from the image with given minimum size restriction
    :param pointcloud_box: scene bounding box
    :param min_size: minimum size of patch in pixels
    :param resolution: Size of pixel in meters

    :return: Random bounding box patch inside bounding box with minimum size ensured
    """
    new_box0_limit = pointcloud_box[2] - pointcloud_box[0] - min_size[0] * resolution
    new_box1_limit = pointcloud_box[3] - pointcloud_box[1] - min_size[1] * resolution
    new_box0 = np.random.rand() * new_box0_limit + pointcloud_box[0]
    new_box1 = np.random.rand() * new_box1_limit + pointcloud_box[1]
    new_box2_limit = pointcloud_box[2] - new_box0 - min_size[0] * resolution
    new_box3_limit = pointcloud_box[3] - new_box1 - min_size[1] * resolution
    new_box2 = np.random.rand() * new_box2_limit + min_size[0] * resolution + new_box0
    new_box3 = np.random.rand() * new_box3_limit + min_size[1] * resolution + new_box1

    return np.array([new_box0, new_box1, new_box2, new_box3])


def generate_intensity_map(pc_input: np.array,
                           pointcloud_box: np.array,
                           out_path: str,
                           transformation=None,
                           resolution=STANDARD_RESOLUTION):
    """
    Projects pointcloud onto BEV 2d image and saves it in corresponding file
    Each pixel corresponds to size resolutionxresolution and contains IDW of closest neighbour intensities
    Creates intensity mask for filtering out ground truth info
    :param pc_input: Point cloud size N x 4, [x, y, z, intensity value]
    :param pointcloud_box: Bounding box [xmin, ymin, xmax, ymax]
    :param out_path: Output path of the image
    :param transformation: Transformation to the minimum bounding box of sensor info
    :param resolution: Size of pixel in meters

    :return: Intensity mask for filtering out grounding truth, saves intensity image as greyscale image
    """
    pc = np.copy(pc_input)
    # Generate shape of image and init with zeros
    W = int((pointcloud_box[2] - pointcloud_box[0]) / resolution) + 1
    H = int((pointcloud_box[3] - pointcloud_box[1]) / resolution) + 1
    image = np.zeros((W, H, 1), dtype='float32')
    image_mask = np.zeros((W, H, 1), dtype='float32')
    counters = np.zeros((W, H), dtype='float32')
    # normalize coordinates to new reference view from corner of box
    pc[:, 0] -= pointcloud_box[0]
    pc[:, 1] -= pointcloud_box[1]
    # Convert x and y coordinate to cell positions
    pc[:, 0] = (pc[:, 0] / resolution)
    pc[:, 1] = (pc[:, 1] / resolution)
    # Filter bigger than bound and less than zero
    pc = pc[pc[:, 0] < W]
    pc = pc[pc[:, 0] >= 0]
    pc = pc[pc[:, 1] < H]
    pc = pc[pc[:, 1] >= 0]

    # Generate kdtree for nn lookup
    kdtree = cKDTree(pc[:, 0:2])
    flatten_pointgrid = np.mgrid[0:W, 0:H].T.reshape(-1, 2)
    # Query pixel grid points and calculate idw
    distances, indexes = kdtree.query(flatten_pointgrid, k=9, eps=0.2, p=2)
    distances = np.divide(1, distances, out=np.zeros_like(distances), where=distances != 0)
    distances_sums = np.sum(distances, axis=1, keepdims=True)
    distances_sums[distances_sums == 0] = 1
    distances /= distances_sums
    indexes[indexes == len(pc)] = 0
    for i in tqdm(range(len(flatten_pointgrid)), disable=TQDM_DISABLE):
        intensity_value = np.mean(distances[i] * pc[indexes[i], 3])
        image[flatten_pointgrid[i, 0], flatten_pointgrid[i, 1], 0] = intensity_value
    # Generate intensity mask with higher upper bound
    distances, indexes = kdtree.query(flatten_pointgrid, k=2, eps=0.1, p=2, distance_upper_bound=20)
    distances = np.divide(1, distances, out=np.zeros_like(distances), where=distances != 0)
    indexes[indexes == len(pc)] = 0
    for i in tqdm(range(len(flatten_pointgrid)), disable=TQDM_DISABLE):
        intensity_value = np.mean(distances[i] * pc[indexes[i], 3])
        image_mask[flatten_pointgrid[i, 0], flatten_pointgrid[i, 1], 0] = intensity_value

    intensity_mask = image_mask > 0
    # Normalize image and save it as image
    image = (2 ** PIXEL_ORDER - 1) * (image - np.min(image)) / (np.max(image) - np.min(image))
    if transformation is not None:
        image = transformation(image)
    cv.imwrite(out_path, image.astype(np.uint16 if PIXEL_ORDER == 16 else np.uint8))

    del pc, image, image_mask, counters

    return intensity_mask


def generate_gradient_magnitude(pc_input: np.array,
                                pointcloud_box: np.array,
                                out_path: str,
                                transformation=None,
                                resolution=STANDARD_RESOLUTION):
    """
    Calculate magnitude of projected BEV z-axis gradient and saves it in file
    Each pixel corresponds to size resolutionxresolution and contains magnitude of gradient calculated by
    obtaining Sobel derivatives first
    :param pc_input: Point cloud size N x 4, [x, y, z, intensity value]
    :param pointcloud_box: Bounding box [xmin, ymin, xmax, ymax]
    :param out_path: Output path of the image
    :param transformation: Transformation to the minimum bounding box of sensor info
    :param resolution: Size of pixel

    :return: None, saves file as greyscale image
    """
    pc = np.copy(pc_input)
    # Generate shape of image and init with zeros
    W = int((pointcloud_box[2] - pointcloud_box[0]) / resolution) + 1
    H = int((pointcloud_box[3] - pointcloud_box[1]) / resolution) + 1
    maxzaxis = np.zeros((W, H, 1), dtype='float32')
    counters = np.zeros((W, H), dtype='float32')
    # normalize coordinates to new reference view from corner of box
    pc[:, 0] -= pointcloud_box[0]
    pc[:, 1] -= pointcloud_box[1]
    # Convert x and y coordinate to cell positions
    pc[:, 0] = (pc[:, 0] / resolution)
    pc[:, 1] = (pc[:, 1] / resolution)
    # Filter bigger than bound and less than zero
    pc = pc[pc[:, 0] < W]
    pc = pc[pc[:, 0] >= 0]
    pc = pc[pc[:, 1] < H]
    pc = pc[pc[:, 1] >= 0]
    # Generate max zaxis values for every pixel separately and iteratively
    for point in tqdm(pc, disable=TQDM_DISABLE):
        value = point[2]
        x = int(point[0])
        y = int(point[1])
        maxzaxis[x, y, 0] = max(maxzaxis[x, y, 0], value)
        counters[x, y] += 1
    # Cleanup negative values
    maxzaxis[maxzaxis < 0] = 0
    # Generate Sobel derivatives
    xsobel = cv.Sobel(maxzaxis, ddepth=-1, dx=1, dy=0, ksize=11)
    ysobel = cv.Sobel(maxzaxis, ddepth=-1, dx=0, dy=1, ksize=11)
    # Calculate magnitude
    image = ((xsobel ** 2 + ysobel ** 2) ** 1/2)

    # Normalize
    image = (2 ** PIXEL_ORDER - 1) * (image - np.min(image)) / (np.max(image) - np.min(image))
    if transformation is not None:
        image = transformation(image)
    cv.imwrite(out_path, image.astype(np.uint16 if PIXEL_ORDER == 16 else np.uint8))

    del pc, image, maxzaxis, counters

    return


def render_map_as_smooth_line_list(
                        nusc: NuScenes,
                        scene_token: str,
                        data_path: str = '../data/sets/nuscenes',
                        box_coordinates: np.array = None,
                        layer_names: List[str] = None) -> List[LineString]:
    """
    Renders map in the limit of one scene as separate combination of lines, and smoothes lines together if they touch
    :param nusc: The NuScenes instance to load the image from.
    :param scene_token: Scene token of map to render as point cloud
    :param data_path: Path to map files
    :param box_coordinates: Box coordinates of map to extract
    :param layer_names: The names of the layers to render, e.g. ['lane'].
        If set to None, the recommended setting will be used.
    :return: list of shapely lines
    """

    # Load correct NusceneMap.
    scene_record = nusc.get('scene', scene_token)
    log_record = nusc.get('log', scene_record['log_token'])
    log_location = log_record['location']
    nusc_map = NuScenesMap(dataroot=data_path, map_name=log_location)

    # Default layers.
    if layer_names is None:
        layer_names = ['drivable_area']

    # Retrieve the current map.
    records_in_patch = nusc_map.get_records_in_patch(box_coordinates, layer_names, 'intersect')
    # Create polygon box
    polygon_box = box(box_coordinates[0], box_coordinates[1], box_coordinates[2], box_coordinates[3])

    # Retrieve every polygon, from polygon retrieve lines separately, intersect with box
    lines_array = []

    for layer_name in layer_names:
        for token in tqdm(records_in_patch[layer_name], disable=TQDM_DISABLE):
            record = nusc_map.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = record['polygon_tokens']
            else:
                polygon_tokens = [record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nusc_map.extract_polygon(polygon_token)
                boundary = polygon.boundary
                # Process multilines/lines
                if boundary.type == 'MultiLineString':
                    for line in boundary:
                        line = line.intersection(polygon_box)
                        # Process intersection
                        if not line.is_empty:
                            if line.type == 'MultiLineString':
                                for lin in line:
                                    lines_array.append(lin)
                            elif line.type == 'LineString':
                                lines_array.append(line)
                else:
                    boundary = boundary.intersection(polygon_box)
                    # Process intersection
                    if not boundary.is_empty:
                        if boundary.type == 'MultiLineString':
                            for line in boundary:
                                lines_array.append(line)
                        elif boundary.type == 'LineString':
                            lines_array.append(boundary)

    lines_array = smooth_line_list(lines_array)

    return lines_array


def smooth_line_list(lines_array: List[LineString]) -> List[LineString]:
    """
    Smoothes line list, so separate boundaries are joined together if they touch at first or last vertex
    :param lines_array: Input lines array
    :return: array of lines with lines joined if they touch at first or last vertex
    """
    # Check if separate line touch with each other, if they do conjugate them
    any_touches = True
    # While loop for proof
    while any_touches:
        any_touches = False
        new_lines_array = []
        consumed_lines_indexes = []
        # Iterate over all lines
        for i, line in enumerate(lines_array):
            # Check if line was consumed by other lines
            if i in consumed_lines_indexes:
                continue
            combined_line = LineString(line)
            # Check connection with other lines
            for j, another_line in enumerate(lines_array):
                # Cant touch itself
                if j == i:
                    continue
                else:
                    # If touches at first or last point combine them with first or last point of other line combine them
                    # (total 4 possible cases)
                    x0, y0 = combined_line.coords[0]
                    x1, y1 = combined_line.coords[-1]
                    x2, y2 = another_line.coords[0]
                    x3, y3 = another_line.coords[-1]
                    if ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 < MIN_TOUCH_DISTANCE:  # Last with first
                        combined_line = LineString(list(combined_line.coords) + list(another_line.coords[1:]))
                        any_touches = True
                        consumed_lines_indexes.append(j)
                    elif ((x0 - x2) ** 2 + (y0 - y2) ** 2) ** 0.5 < MIN_TOUCH_DISTANCE:  # First with first
                        combined_line = LineString(list(combined_line.coords)[::-1] + list(another_line.coords[1:]))
                        any_touches = True
                        consumed_lines_indexes.append(j)
                    elif ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 0.5 < MIN_TOUCH_DISTANCE:  # Last with Last
                        combined_line = LineString(list(combined_line.coords) + list(another_line.coords[:-1])[::-1])
                        any_touches = True
                        consumed_lines_indexes.append(j)
                    elif ((x0 - x3) ** 2 + (y0 - y3) ** 2) ** 0.5 < MIN_TOUCH_DISTANCE:  # First with Last
                        combined_line = LineString(list(combined_line.coords)[::-1] + list(another_line.coords[:-1])[::-1])
                        any_touches = True
                        consumed_lines_indexes.append(j)
            # Will append new combined line, or either will append the original unchanged line
            new_lines_array.append(combined_line)

        lines_array = new_lines_array

    return lines_array


def generate_endpoints_map(lines_list: List[LineString],
                           pointcloud_box: np.array,
                           out_path: str,
                           transformation=None,
                           resolution=STANDARD_RESOLUTION):
    """
    Calculate endpoints for every boundary line and save them as image/numpy file
    :param lines_list: List containing all boundary lines
    :param pointcloud_box: Bounding box [xmin, ymin, xmax, ymax]
    :param out_path: Path to output image file
    :param transformation: Transformation to the minimum bounding box of sensor info
    :param resolution: Size of pixel
    :return: None, saves file as greyscale image
    """
    W = int((pointcloud_box[2] - pointcloud_box[0]) / resolution) + 1
    H = int((pointcloud_box[3] - pointcloud_box[1]) / resolution) + 1
    image = np.zeros((W, H, 1), dtype='float32')
    for line in tqdm(lines_list, disable=TQDM_DISABLE):
        x_start, y_start = line.coords[0]
        x_end, y_end = line.coords[-1]
        image[int((x_start - pointcloud_box[0]) / resolution), int((y_start - pointcloud_box[1]) / resolution), 0] = 1
        image[int((x_end - pointcloud_box[0]) / resolution), int((y_end - pointcloud_box[1]) / resolution), 0] = 1
    # Normalize
    image = (2 ** PIXEL_ORDER - 1) * (image - np.min(image)) / (np.max(image) - np.min(image))
    if transformation is not None:
        image = transformation(image)
    cv.imwrite(out_path, image.astype(np.uint16 if PIXEL_ORDER == 16 else np.uint8))

    del image

    return


def generate_vector_distance_and_direction_map(lines_list: List[LineString],
                                               pointcloud_box: np.array,
                                               out_path_list: List[str],
                                               transformation=None,
                                               resolution=STANDARD_RESOLUTION,
                                               intensity_mask=None):
    """
    Calculate normalized vector of distance and distance from every point to closest road boundary
    :param lines_list: List containing all boundary lines
    :param pointcloud_box: Bounding box [xmin, ymin, xmax, ymax]
    :param out_path_list: List of pathes to output files
    :param transformation: Transformation to the minimum bounding box of sensor info
    :param resolution: Size of pixel
    :param intensity_mask: Intensity mask for filtering out reverse distance ground truth
    :return: None, saves file as numpy array and rgb image (red channel is zero)
    """
    multiline = MultiLineString(lines_list)
    # Generate shape of image and init with zeros
    W = int((pointcloud_box[2] - pointcloud_box[0]) / resolution) + 1
    H = int((pointcloud_box[3] - pointcloud_box[1]) / resolution) + 1
    image_distance = np.zeros((W, H, 1), dtype='float32')

    image_vector = np.zeros((W, H, 2), dtype='float32')
    # Generate new point for each cell and move it to lines perspective,
    # then calculate nearest point on multiline string
    # Use it to calculate the normalized vector
    for ix, iy in tqdm(np.ndindex((image_vector.shape[0], image_vector.shape[1])),
                       total=image_vector.shape[0] * image_vector.shape[1],
                       disable=TQDM_DISABLE):
        point = Point(ix * resolution + pointcloud_box[0], iy * resolution + pointcloud_box[1])
        xstart, ystart = point.coords[0]
        nearest = nearest_points(point, multiline)[1]
        xend, yend = nearest.coords[0]
        vector_length = ((xend - xstart) ** 2 + (yend - ystart) ** 2) ** 0.5
        image_distance[ix, iy, 0] = 1 / max(vector_length, resolution)
        image_vector[ix, iy] = np.array([(xend - xstart) / vector_length, (yend - ystart) / vector_length])

    # Save inverse distance image
    image_distance = (2 ** PIXEL_ORDER - 1) * (image_distance - np.min(image_distance)) /\
                     (np.max(image_distance) - np.min(image_distance))
    image_distance = np.multiply(image_distance, intensity_mask)
    if transformation is not None:
        image_distance = transformation(image_distance)
    cv.imwrite(out_path_list[0], image_distance.astype(np.uint16 if PIXEL_ORDER == 16 else np.uint8))

    image_vector_saved = np.zeros((W, H, 3), dtype='float32')
    # Save normalized info in 2 channels
    image_vector_saved[:, :, 0:2] = (2 ** PIXEL_ORDER - 1) * (image_vector - np.min(image_vector)) /\
                                    (np.max(image_vector) - np.min(image_vector))
    # incode signs values as the third, red channel
    image_vector_saved[:, :, 2] = (image_vector[:, :, 0] > 0) * (2 ** (PIXEL_ORDER - 2)) +\
                                  (image_vector[:, :, 1] > 0) * (2 ** (PIXEL_ORDER - 1))
    if transformation is not None:
        image_vector_saved = transformation(image_vector_saved)
    cv.imwrite(out_path_list[1], image_vector_saved.astype(np.uint16 if PIXEL_ORDER == 16 else np.uint8))

    del image_vector, image_distance, image_vector_saved

    return


def generate_transformation_min_box(poserecord_history: np.array,
                                    pointcloud_box: np.array,
                                    resolution=STANDARD_RESOLUTION) -> Any:
    """
    Calculate transformation for cutting minimum area of image containing information from useful range of lidar
    :param poserecord_history: List containing all locations of car poses
    :param pointcloud_box: Bounding box [xmin, ymin, xmax, ymax]
    :param resolution: Size of pixel
    :return: Transformation function with preset parameters
    """
    # Create test image
    W = int((pointcloud_box[2] - pointcloud_box[0]) / resolution) + 1
    H = int((pointcloud_box[3] - pointcloud_box[1]) / resolution) + 1
    image_test = np.zeros((W, H, 1), dtype='float32')
    # Draw circles containing useful info
    for poserecord in poserecord_history:
        cv.circle(image_test, (int((poserecord[1] - pointcloud_box[1]) / resolution),
                               int((poserecord[0] - pointcloud_box[0]) / resolution)),
                  int(BOX_MAX / resolution), 255, 1)
    points = np.transpose(np.flip(np.where(image_test == 255)[:2], axis=0))
    # Check if transformation possible
    if points.size == 0:
        return None, None
    # Extract minimum rectangle
    rect = cv.minAreaRect(points)
    # Create perspective transform
    box_ = cv.boxPoints(rect)
    box_ = np.int0(box_)
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box_.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    # Return partial function with preset parameters
    return functools.partial(cv.warpPerspective, M=M, dsize=(width, height), borderMode=cv.BORDER_REPLICATE), \
           functools.partial(cv.warpPerspective, M=M, dsize=(width, height))


def main(nusc: NuScenes, use_as_function=False, scene_name='scene-0061', gt_gen=True,
         patch_gen=0, patch_gen_only=0, min_box=0, out_dir='input/misc'):
    """
    Main function that generates all the data from one single Nuscenes scene
    :param nusc: The NuScenes instance to load the image from.
    :param use_as_function: whether to use as a function (see mass_stage1_input_generation.py)
    :param scene_name: Name of a scene, e.g. scene-0061
    :param gt_gen: Whether to generate ground truth from map
    :param patch_gen: Number of additional patches to generate
    :param patch_gen_only: Whether to generate patches only
    :param min_box: Whether to cut image boxes to effective range of lidar
    :param out_dir: General output folder for the dataset
    :return:
    """
    if not use_as_function:
        # Read input parameters
        parser = argparse.ArgumentParser(description='Generate input for neural network stage 1'
                                                     ' and ground truth values.',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--scene_name', default='scene-0061', type=str, help='Name of a scene, e.g. scene-0061')
        parser.add_argument('--gt_gen', default=1, type=int, help='Whether to generate ground truth from map')
        parser.add_argument('--patch_gen', default=0, type=int, help='Number of additional patches to generate')
        parser.add_argument('--patch_gen_only', default=0, help='Whether to generate patches only')
        parser.add_argument('--min_box', default=0, help='Whether to cut image boxes to effective range of lidar')
        parser.add_argument('--ver', default='v1.0-mini', type=str, help='Version of dataset')
        parser.add_argument('--out_dir', default=osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                                                          'input', 'mini_set'), type=str, help='Output folder for the whole dataset, scenes in subfolders')

        args = parser.parse_args()
        scene_name = args.scene_name
        gt_gen = bool(args.gt_gen)
        version = args.ver
        patch_gen = args.patch_gen
        patch_gen_only = bool(args.patch_gen_only)
        min_box = bool(args.min_box)
        out_dir = os.path.expanduser(args.out_dir)
        nusc = NuScenes(version=version, dataroot=osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
                                                           'data', 'sets', 'nuscenes'), verbose=False)
    else:
        out_dir = os.path.expanduser(out_dir)

    out_dir = osp.join(out_dir, scene_name)

    # Create output folder
    if not out_dir == '' and not osp.isdir(out_dir):
        os.makedirs(out_dir)
    # Check if scene exist
    scene_tokens = [s['token'] for s in nusc.scene if s['name'] == scene_name]
    assert len(scene_tokens) == 1, 'Error: Invalid scene %s' % scene_name
    # Export scene data from dataset
    print('Processing scene', scene_name)
    print('=> Extracting pointcloud from dataset')
    pc, pointcloud_box, poserecord_history = export_scene_pointcloud(nusc, scene_tokens[0])
    # Generate transformations for cutting information outside of effective LIDAR range
    if min_box:
        reflect_transformation, transformation = generate_transformation_min_box(poserecord_history, pointcloud_box)
    else:
        reflect_transforamtion, transformation = None, None
    # Generate main scene
    if not patch_gen_only:
        # Generate intensity BEV image
        print('=> Generating intensity map')
        intensity_mask = \
            generate_intensity_map(pc, pointcloud_box,
                                   osp.join(out_dir, '%s_intensity.png' % scene_name), transformation=reflect_transformation)
        # Generate gradient BEV image
        print('=> Generating gradient magnitude')
        generate_gradient_magnitude(pc, pointcloud_box,
                                    osp.join(out_dir, '%s_gradient.png' % scene_name), transformation=transformation)
        # Generate ground truth BEV images
        if gt_gen:
            print('=> Generating road boundaries geometries')
            lines_list = render_map_as_smooth_line_list(nusc, scene_tokens[0], box_coordinates=pointcloud_box)
            print('=> Generating truth map for road endpoints to road boundaries')
            generate_endpoints_map(lines_list, pointcloud_box,
                                   osp.join(out_dir, '%s_endpoints.png' % scene_name), transformation=transformation)
            print('=> Generating truth map for inverse distance and normalized vector directions to road boundaries')
            generate_vector_distance_and_direction_map(lines_list, pointcloud_box,
                                                       [osp.join(out_dir, '%s_distance.png' % scene_name),
                                                        osp.join(out_dir, '%s_vector.png' % scene_name)],
                                                       transformation=transformation,
                                                       intensity_mask=intensity_mask)
    # Generate patches
    if patch_gen > 0:
        # Patches are stored in the separate sub-folder
        patch_dir = osp.join(out_dir, 'stage1_patches')
        if not patch_dir == '' and not osp.isdir(patch_dir):
            os.makedirs(patch_dir)

        print('=> Generating patches')
        for i in tqdm(range(patch_gen), disable=TQDM_DISABLE):
            # Generate patch boxes
            patch_pointcloud_box = generate_random_patch(pointcloud_box, min_size=((pointcloud_box[2] - pointcloud_box[0]) / (2 * STANDARD_RESOLUTION),
                                                                                   (pointcloud_box[3] - pointcloud_box[1]) / (2 * STANDARD_RESOLUTION)))
            lines_list = render_map_as_smooth_line_list(nusc, scene_tokens[0], box_coordinates=patch_pointcloud_box)
            # Generate transformations for cutting information outside of effective LIDAR range
            if min_box:
                reflect_transformation, transformation = generate_transformation_min_box(poserecord_history, patch_pointcloud_box)
            else:
                reflect_transformation, transformation = None, None
            # If box holding viable data in this patch exist, generate all the data for this patch
            if lines_list and (min_box is False or transformation is not None):
                intensity_mask = \
                    generate_intensity_map(pc, patch_pointcloud_box, osp.join(patch_dir, '%s_%i_intensity.png' % (scene_name, i)), transformation=reflect_transformation)
                generate_gradient_magnitude(pc, patch_pointcloud_box, osp.join(patch_dir, '%s_%i_gradient.png' % (scene_name, i)), transformation=transformation)
                if gt_gen:
                    generate_endpoints_map(lines_list, patch_pointcloud_box, osp.join(patch_dir, '%s_%i_endpoints.png' % (scene_name, i)), transformation=transformation)
                    generate_vector_distance_and_direction_map(lines_list, patch_pointcloud_box,
                                                               [osp.join(patch_dir, '%s_%i_distance.png' % (scene_name, i)),
                                                                osp.join(patch_dir, '%s_%i_vector.png' % (scene_name, i))],
                                                               transformation=transformation,
                                                               intensity_mask=intensity_mask)

    return


if __name__ == '__main__':
    main(None)

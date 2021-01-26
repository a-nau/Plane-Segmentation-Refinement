import itertools
import logging
import os
import time
import warnings

import cv2
import numpy as np

from config import GlobalConfig as GlobalConfig
from refiner.image_processing.draw import draw_lines_on_image, draw_lines_rho_theta_on_image
from utils.image_modification import get_colored_image

cfg = GlobalConfig.get_config()
logger = logging.getLogger(__name__)


def compute_line_intersection(line1, line2):
    # See https://stackoverflow.com/a/20677983
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if np.any(div == 0):
        return np.inf, np.inf

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]


def get_line_image_coordinates(image, line):
    rho = line[0]
    theta = line[1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 5000 * (-b))
    y1 = int(y0 + 5000 * (a))
    x2 = int(x0 - 5000 * (-b))
    y2 = int(y0 - 5000 * (a))
    intersections = []
    if theta != 0:
        intersections.append(
            compute_line_intersection(
                [[x1, y1], [x2, y2]],
                [[0, 0], [image.shape[1] - 1, 0]]
            )
        )
        intersections.append(
            compute_line_intersection(
                [[x1, y1], [x2, y2]],
                [[image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]]
            )
        )
    if theta != np.pi * 0.5:
        intersections.append(
            compute_line_intersection(
                [[x1, y1], [x2, y2]],
                [[0, 0], [0, image.shape[0] - 1]]
            )
        )
        intersections.append(
            compute_line_intersection(
                [[x1, y1], [x2, y2]],
                [[image.shape[1] - 1, 0], [image.shape[1] - 1, image.shape[0] - 1]]
            )
        )
    intersections = [ip for ip in intersections if
                     (np.minimum(ip[0], ip[1]) >= 0 and ip[0] <= image.shape[1] and ip[1] <= image.shape[0])]
    if len(intersections) == 2:
        return np.array([intersections[0], intersections[1]], np.int)
    else:
        return None


def get_lines_from_edges(edge_img, minLength=100, rho=1, theta=np.pi / 180, maxLineGap=20, prob=False):
    lines = []
    if prob:
        hough_lines = cv2.HoughLinesP(edge_img, rho, theta, 250, minLength, maxLineGap)
        if not hough_lines is None:
            lines = [[i[:2], i[2:]] for i in hough_lines.reshape(-1, 4)]
    else:
        hough_lines = cv2.HoughLines(edge_img, rho, theta, minLength)
        if not hough_lines is None:
            lines = [i[0] for i in hough_lines]

    return lines


def detect_lines(binary_image: np.ndarray,
                 min_line_number=0,
                 min_edge_length=100,
                 cluster_lines=True,
                 probabilistic=True,
                 output_directory=None):
    final_lines = []
    lines_normal_form = get_lines_from_edges(binary_image, int(min_edge_length), 1, np.pi / 180, prob=probabilistic)
    if output_directory is not None:  # draw lines of full length
        line_image = draw_lines_rho_theta_on_image(get_colored_image(binary_image), lines_normal_form,
                                                   color=tuple([255, 0, 0]), thickness=1)
        if cfg.visualization_dict['hough_lines']:
            cv2.imwrite(
                os.path.join(output_directory,
                             "image_hough_all_{}.png".format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))),
                line_image)
    if len(lines_normal_form) > 0:
        while len(lines_normal_form) < min_line_number:
            min_edge_length = min_edge_length * 0.8
            lines_normal_form = get_lines_from_edges(binary_image, int(min_edge_length), 1, np.pi / 900)

        if cluster_lines and not probabilistic:
            lines_normal_form = cluster_lines_and_average(lines_normal_form)  # result in normal form (rho, theta)
            lines_point_form = [convert_normal_to_point_form_of_line(line[0], line[1]) for
                                line in lines_normal_form]
            final_lines = find_line_end_points(binary_image, lines_point_form)  # result in point form (p0, p1)
            if output_directory is not None:  # draw lines of full length
                line_image = draw_lines_rho_theta_on_image(get_colored_image(binary_image), lines_normal_form,
                                                           color=tuple([255, 0, 0]), thickness=6)
                line_image = draw_lines_on_image(line_image, final_lines, color=tuple([0, 255, 0]), thickness=4)
                if cfg.visualization_dict['hough_summary']:
                    cv2.imwrite(
                        os.path.join(output_directory,
                                     "image_hough_choice_{}.png".format(
                                         time.strftime("%Y%m%d_%H%M%S", time.localtime()))),
                        line_image)
    return final_lines


def cluster_lines_and_average(lines):
    from refiner.clustering.dbscan import dbscan_with_points
    cluster_size = 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cluster = dbscan_with_points(np.array(lines), eps=0.15, min_samples=cluster_size)
    new_lines = []
    for key, val in cluster.items():
        if len(val) > 0:
            smoothed = np.mean(val, axis=0)
            if not np.isnan(smoothed).any():
                new_lines.append(smoothed)
    return new_lines


def find_line_end_points(binary_image, lines, close_to_current=False):
    new_lines = []
    logger.debug("Checking {} lines...".format(len(lines)))
    for i, line0 in enumerate(lines):
        possible_points = []
        subset_lines = lines.copy()
        del subset_lines[i]
        for j, line1 in enumerate(subset_lines):
            intersection = compute_line_intersection(line0, line1)
            if point_close_to_mask(intersection, binary_image, size=40):
                if close_to_current:
                    if check_if_intersection_close_to_line_points(intersection, line0, line1):
                        possible_points.append(intersection)
                else:
                    possible_points.append(intersection)
        if len(possible_points) == 2:
            new_lines.append(possible_points)
        elif len(possible_points) > 2:
            for line in itertools.combinations(possible_points, 2):
                if line_on_mask(line, binary_image):
                    new_lines.append(line)
    return new_lines


def find_line_end_points_detailed(binary_image, lines):
    from refiner.combined.combined_appraoch import find_line_with_maximum_reward

    new_lines = []
    logger.debug("Checking {} lines...".format(len(lines)))
    for i, line0 in enumerate(lines):
        possible_points = []
        subset_lines = lines.copy()
        del subset_lines[i]
        for j, line1 in enumerate(subset_lines):
            intersection = compute_line_intersection(line0, line1)
            close_point_idx = np.argmin(np.linalg.norm(np.array(intersection).reshape(-1, 2)
                                                       - np.array(line0).reshape(-1, 2), axis=1))
            close_point = line0[close_point_idx]
            if point_close_to_mask(intersection, binary_image, size=50):
                if check_if_intersection_close_to_line_points(intersection, line0, line1, threshold=50):
                    possible_points.append(list(intersection))
                elif line_on_mask([close_point, intersection], binary_image, width=1, iou_threshold=0.75):
                    possible_points.append(list(intersection))
        if len(possible_points) == 2:
            new_lines.append(list(possible_points))
        elif len(possible_points) > 2:
            line = find_line_with_maximum_reward(list(itertools.combinations(possible_points, 2)), binary_image)
            new_lines.append(list(line))
        else:
            new_lines.append(list(line0))
    return new_lines


def point_close_to_mask(point, mask, size=20):
    if np.linalg.norm(point) > 10 ** 6:
        return False
    point_image = np.zeros_like(mask).astype(np.int8)
    point_image = cv2.circle(point_image, tuple(int(p) for p in point), size, tuple([255]), -1)
    intersection_image = cv2.bitwise_and(mask, mask, mask=point_image)
    return np.sum(intersection_image) > 0


def line_on_mask(line, mask, width=2, iou_threshold=0.6):
    iou = compute_iou_mask_and_line(line, mask, width)
    return iou > iou_threshold


def compute_iou_mask_and_line(line, mask, width):
    line_image = np.zeros_like(mask).astype(np.uint8)
    line_image = cv2.line(line_image,
                          tuple(int(p) for p in line[0]),
                          tuple(int(p) for p in line[1]),
                          tuple([255]),
                          width)
    intersection_image = cv2.bitwise_and(mask, mask, mask=line_image)
    iou = (np.sum(intersection_image) / np.max(intersection_image)) / (np.sum(line_image) / np.max(line_image))
    return iou


def convert_normal_to_point_form_of_line(rho, theta):
    points = []
    for x in [0, 1920]:
        points.append(np.array([x, (rho - x * np.cos(theta)) / np.sin(theta)]))
    return points


def convert_point_to_normal_form_of_line(point0, point1):
    m = (point1[1] - point0[1]) / (point1[0] - point0[0])
    x_intercept = point0[1] - (m * point0[0])
    if m == 0:
        y_intercept = np.inf
    else:
        y_intercept = - x_intercept / m

    theta = np.arctan(y_intercept / x_intercept)
    rho = y_intercept * np.cos(theta)

    return [rho, theta]


def check_if_intersection_close_to_line_points(intersection, line0, line1, threshold=50):
    dists = []
    for line in [line0, line1]:
        dists.append(np.min([np.linalg.norm(np.array(intersection) - np.array(line[i])) for i in [0, 1]]))
    return np.all(np.array(dists) < threshold)

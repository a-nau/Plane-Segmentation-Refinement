import itertools

import cv2
import numpy as np
import logging

from refiner.clustering.dbscan import dbscan_with_points
from refiner.hough_transform.hough_transformation import convert_point_to_normal_form_of_line, \
    compute_iou_mask_and_line
from utils.other import get_depth_of_nested_list

from config import GlobalConfig as GlobalConfig
cfg = GlobalConfig.get_config()
logger = logging.getLogger(__name__)


def cluster_lines_together_and_choose_best(lines, gray_scaled_img):
    lines_rho_theta = [convert_point_to_normal_form_of_line(line[0], line[1]) for line in lines]
    clustered_lines = find_clustered_lines(lines_rho_theta, lines)
    if len(clustered_lines) == 1 and get_depth_of_nested_list(clustered_lines) <= 3:
        clustered_lines = [clustered_lines]  # list comprehension has issues without this
    best_lines = [find_best_line(l, gray_scaled_img) for l in clustered_lines]
    return best_lines


def find_clustered_lines(lines_rho_theta, lines):
    if len(lines_rho_theta) == 1:
        return lines
    try:
        cluster = dbscan_with_points(np.array(lines_rho_theta), eps=0.15, min_samples=1)
    except ValueError:
        cluster = dict()
    clustered_lines = []
    for key, val in cluster.items():
        curr_lines = val.tolist()
        clustered_line = []
        for line in curr_lines:
            round_val = 6
            idx = ["".join([str(round(i, round_val)) for i in l]) for l in lines_rho_theta].index(
                "".join([str(round(i, round_val)) for i in line]))  # find index
            clustered_line.append(list(lines[idx]))
        clustered_lines.append(clustered_line)
    return clustered_lines


def find_best_line(lines, gray_scaled_img):
    from sklearn.cluster import KMeans

    points = np.array(sum(lines, [])).reshape(-1, 2)
    initial_line = np.array(lines[0]).reshape(2, 2)  # take one as start for clustering
    kmeans = KMeans(n_clusters=2, random_state=0, init=initial_line, n_init=1).fit(points)
    start_points = points[kmeans.labels_ > 0]
    end_points = points[kmeans.labels_ < 1]
    start_point_cluster_center = kmeans.cluster_centers_[1]
    end_point_cluster_center = kmeans.cluster_centers_[0]
    # sample points close to cluster centers and add them
    start_points_around_cluster_center = sample_point_from_intersection_of_mask_and_point(start_point_cluster_center,
                                                                                          gray_scaled_img, 20,
                                                                                          cfg.n_random_points)
    end_points_around_cluster_center = sample_point_from_intersection_of_mask_and_point(end_point_cluster_center,
                                                                                        gray_scaled_img, 20,
                                                                                        cfg.n_random_points)
    if len(start_points_around_cluster_center) > 0:
        start_points = np.vstack((start_points, start_points_around_cluster_center))
    if len(end_points_around_cluster_center) > 0:
        end_points = np.vstack((end_points, end_points_around_cluster_center))
    new_lines = list(itertools.product(start_points, end_points))
    best_points = find_line_with_maximum_reward(new_lines, gray_scaled_img)
    return best_points


def find_line_with_maximum_reward(line_candidates, gray_scaled_img):
    min_iou = 0.9
    best_score = 0
    best_points = line_candidates[0]
    line_candidates_array = np.array(line_candidates).reshape(-1, 4)
    max_dist = np.max(np.linalg.norm(line_candidates_array[:, 0:2] - line_candidates_array[:, 2:], axis=1))
    for p0, p1 in line_candidates:
        iou = compute_iou_mask_and_line([p0, p1], gray_scaled_img, 1)
        dist = np.linalg.norm(np.array(p0) - np.array(p1))
        weight_iou = 0.5
        score = weight_iou * iou + (1 - weight_iou) * dist / max_dist  # reward function
        if iou > min_iou and score > best_score:
            best_score = score
            best_points = [p0, p1]
    return best_points


def sample_point_from_intersection_of_mask_and_point(point, mask, size=20, n_samples=10):
    import random
    point_image = np.zeros_like(mask).astype(np.int8)
    point_image = cv2.circle(point_image, tuple(int(p) for p in point), size, tuple([255]), -1)
    intersection_image = cv2.bitwise_and(mask, mask, mask=point_image)
    masked_points = np.where(intersection_image > 0)
    if len(masked_points) > 0:
        X = np.column_stack(tuple((masked_points[1], masked_points[0]))).tolist()
        return np.array(random.sample(X, min(len(X), n_samples)))

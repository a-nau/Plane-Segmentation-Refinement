import logging
import os

import cv2
import numpy as np

from config import GlobalConfig as GlobalConfig
from refiner.image_processing.draw import draw_corners_on_image, draw_masks_on_image
from refiner.image_processing.evaluation_metrics import compare_iou_of_points_with_mask
from refiner.util.ransac_linear_regression import ransac_linear_regression
from utils.image_modification import scale_image, get_grayscaled_image, get_colored_image
from utils.other import get_unique_rows

cfg = GlobalConfig.get_config()
logger = logging.getLogger(__name__)


def compute_lines_from_edge_candidate_clusters(img_edges, edge_candidates, mask_plane, mask_extract_contour,
                                               mask_number,
                                               output_directory):
    img_edge_candidates = np.copy(img_edges)
    img_lines = np.copy(img_edges)
    lines = []
    for i, (key, val) in enumerate(edge_candidates.items()):
        color = cfg.color_pallet[i]
        val = get_unique_rows(val)
        img_edge_candidates = draw_corners_on_image(val, get_colored_image(img_edges.copy()), color=color, radius=1)
        y = img_edges.shape[0] - val[:, 1].reshape(-1, 1)  # OpenCV CS starts in upper left corner
        X = val[:, 0].reshape(-1, 1)

        prediction = ransac_linear_regression(X, y, draw=False)
        if prediction is not None:
            line_masked_image = find_line_masked_image(prediction, np.copy(img_edges), mask_number, key,
                                                       output_directory)
            line = find_final_line(line_masked_image)
            if line is None:
                logger.debug("Use minimal line because other algorithm didn't work")
                line = [[int(x), int(img_edges.shape[0] - prediction(x.reshape(1, -1)))] for x in
                        [np.min(X), np.max(X)]]
            img_lines = cv2.line(img_lines, tuple(line[0]), tuple(line[1]), color, 3)
            lines.append(line)
    if cfg.visualization_dict['lines']:
        cv2.imwrite(os.path.join(output_directory, "mask_{}_line_image.jpg".format(mask_number)), img_lines)
    output_img = get_colored_image(img_edge_candidates)
    if cfg.visualization_dict['keep_edges']:
        cv2.imwrite(os.path.join(output_directory, "mask_{}_keep_edges.jpg".format(mask_number)), output_img)
    image_masked = draw_masks_on_image(output_img, [mask_plane])
    if cfg.visualization_dict['initial_mask']:
        cv2.imwrite(os.path.join(output_directory, "mask_{}_image.jpg".format(mask_number)), image_masked)
    return lines


def find_line_masked_image(equation, image, n_mask, n_edge, output_directory):
    # mask the line across image
    line_mask = np.zeros_like(get_grayscaled_image(image))
    pt_start = tuple((0, image.shape[0] - equation(np.array([0]).reshape(1, -1))))
    pt_end = tuple((image.shape[1], image.shape[0] - equation(np.array([image.shape[1]]).reshape(1, -1))))
    line_mask = cv2.line(line_mask, pt_start, pt_end, tuple([255]), thickness=15)
    # Overlay line_mask and image
    masked_img = cv2.bitwise_and(image, image, mask=line_mask)
    if cfg.visualization_dict['mask_folder']:
        dir_mask = os.path.join(output_directory, "mask_{}".format(str(n_mask).zfill(2)))
        if not os.path.exists(dir_mask):
            os.mkdir(dir_mask)
        cv2.imwrite(os.path.join(dir_mask, "line_" + str(n_edge) + ".jpg"), masked_img)
    return get_grayscaled_image(masked_img)


def find_final_line(line_masked_image, draw=False):
    from refiner.clustering.dbscan import dbscan_with_masked_image
    clustered_edges = dbscan_with_masked_image(line_masked_image, eps=cfg.clustering_eps,
                                               min_samples=cfg.clustering_min_sample)
    if draw:
        for key, val in clustered_edges.items():
            img_keep_edges = draw_corners_on_image(val, get_colored_image(line_masked_image))
        cv2.imshow("test", scale_image(img_keep_edges, 0.6))
        cv2.waitKey(0)
    cluster = clustered_edges['max']
    X = cluster[:, 0].reshape(-1, 1)
    y = line_masked_image.shape[0] - cluster[:, 1].reshape(-1, 1)  # OpenCV CS starts in upper left corner
    prediction = ransac_linear_regression(X, y)
    if prediction is not None:
        return find_start_and_point_on_mask(X, line_masked_image, prediction)
    else:
        return None


def find_start_and_point_on_mask(X, line_masked_image, prediction):
    X = np.sort(X.reshape(-1, ))

    y_min = None
    i = 0
    while True:
        x_min = X[i].reshape(-1, 1)
        y_min_tmp = line_masked_image.shape[0] - prediction(x_min)[0, 0]
        if y_min_tmp > line_masked_image.shape[0] - 1:
            y_min = y_min_tmp if y_min is None else y_min
            break
        if line_masked_image[int(y_min_tmp), int(x_min)] > 0:  # reverse due to cv2
            y_min = y_min_tmp
            break
        y_min = y_min_tmp
        i += 1

    y_max = None
    i = X.shape[0] - 1
    while True:
        x_max = X[i].reshape(-1, 1)
        y_max_tmp = line_masked_image.shape[0] - prediction(x_max)[0, 0]
        if y_max_tmp > line_masked_image.shape[0] - 1:
            y_max = y_max_tmp if y_max is None else y_max
            break
        if line_masked_image[int(y_max_tmp), int(x_max)] > 0:  # reverse due to cv2
            y_max = y_max_tmp
            break
        y_max = y_max_tmp
        i -= 1
    line = [[int(x_min), int(y_min)],
            [int(x_max), int(y_max)]]
    return line


def find_normal_pointing_in_mask_direction(line, normal, mask, draw=False):
    scale = 500
    directions = [1, -1]
    ious = []
    points_ = []
    for i in directions:
        points = get_points_of_mask_from_line_and_scale(line, normal * i, scale)
        ious.append(compare_iou_of_points_with_mask(points, mask, 255, draw))
        points_.append(points)
    if ious[1] is None and ious[0] is None:
        return None
    elif ious[0] > ious[1] or ious[1] is None:
        idx = 0
    elif ious[0] < ious[1] or ious[0] is None:
        idx = 1
    else:
        return None
    logger.debug("Final IoU is {}".format(ious[idx]))
    return normal * directions[idx]


def compute_normalized_line_directions(line):
    diff = (np.array(line[1]) - np.array(line[0])).reshape(-1, 1)
    vector = diff / np.linalg.norm(diff)
    normal = np.array([-vector[1], vector[0]]).reshape(-1, 1) / np.linalg.norm([-vector[1], vector[0]])
    if normal is None:
        raise ValueError("Normal could not be computed for line {}".format(line))
    return vector, normal


def get_points_of_mask_from_line_and_scale(line, normal, scale):
    pt0 = np.array(line[0])
    pt1 = np.array(line[1])
    pt2 = ((scale * normal).transpose() + line[1]).astype(np.int).reshape(-1, )
    pt3 = ((scale * normal).transpose() + line[0]).astype(np.int).reshape(-1, )
    points = [pt0, pt1, pt2, pt3]
    return points

import logging
import os

import cv2
import numpy as np

from config import GlobalConfig as GlobalConfig
from utils.image_modification import get_grayscaled_image
from utils.other import get_n_unique_rows

cfg = GlobalConfig.get_config()
logger = logging.getLogger(__name__)


def get_edge_candidate_clusters_from_mask(image, mask, n_mask, ksize, output_directory):
    from refiner.clustering.dbscan import dbscan_with_masked_image
    from refiner.image_processing.draw import draw_masks_on_image
    if ksize % 2 == 0:
        raise ValueError("Kernel size must be odd")

    img_masked = draw_masks_on_image(image, [(mask * 255).astype(np.uint8)])
    if cfg.visualization_dict['widened_contour']:
        cv2.imwrite(os.path.join(output_directory, "mask_{}_around_edges.jpg".format(str(n_mask).zfill(2))), img_masked)
    image[mask == 0] = 0
    image = get_grayscaled_image(image)
    clustered_edges = dbscan_with_masked_image(image, eps=cfg.clustering_eps, min_samples=cfg.clustering_min_sample)
    if len(clustered_edges.values()) > 50:
        logger.debug("First run not successful (found {} edges)".format(len(clustered_edges.values())))
        clustered_edges = dbscan_with_masked_image(image, eps=cfg.clustering_eps * 2,
                                                   min_samples=cfg.clustering_min_sample)
    clustered_edges = filter_edges(image, clustered_edges)
    return clustered_edges


def compute_widened_contour_line_around_mask(mask, ksize=37):
    outer_mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)  # enlarge mask
    inner_mask = cv2.GaussianBlur(~mask, (ksize, ksize), 0)  # enlarge inverse mask
    new_mask = np.ones_like(mask)
    new_mask[outer_mask == 0] = 0
    new_mask[inner_mask == 0] = 0
    return new_mask


def filter_edges(masked_image, edges):
    # Compute mask size
    mask = get_grayscaled_image(masked_image) / 255
    mask_size = int(np.sum(mask))

    remove_keys = []
    for key, val in edges.items():
        # Compute edge size
        edge_size = get_n_unique_rows(val)
        if edge_size / mask_size < 0.01:  # remove small ones
            remove_keys.append(key)
        else:  # remove "centered ones" since we are looking for lines
            median = np.median(val, axis=0)
            dists = np.linalg.norm(val - median, axis=1)
            var = np.var(dists)
            threshold = (mask.shape[0] + mask.shape[1]) / 25
            if var < threshold:
                logger.debug(
                    "Removed edges due to small variance in distribution {} < {}".format(round(var, 2),
                                                                                         round(threshold, 2)))
                remove_keys.append(key)

    for key in remove_keys:
        del edges[key]
    return edges

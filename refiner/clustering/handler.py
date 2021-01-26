import logging
import os

import numpy as np

from config import GlobalConfig as GlobalConfig
from refiner.clustering.edge_detection import get_edge_candidate_clusters_from_mask, filter_edges
from refiner.clustering.line_detection import compute_lines_from_edge_candidate_clusters
from refiner.image_processing.corner_detection import get_harris_corners
from utils.image_modification import get_grayscaled_image, get_colored_image
from utils.other import get_n_unique_rows

cfg = GlobalConfig.get_config()
logger = logging.getLogger(__name__)


def get_lines_from_clustering(img_edges, mask_extract_contour, mask_plane, mask_number, output_directory, ksize=51):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    edge_candidate_clusters = get_edge_candidate_clusters_from_mask(
        np.copy(img_edges),
        mask_extract_contour,
        mask_number,
        ksize=ksize,
        output_directory=output_directory
    )
    logger.debug("Found {} edges. Checking if we should split across edges.".format(len(edge_candidate_clusters)))
    edge_candidate_clusters = split_edge_candidate_clusters(
        np.copy(img_edges),
        mask_extract_contour,
        edge_candidate_clusters
    )
    logger.debug("Found {} edges after splitting big edges.".format(len(edge_candidate_clusters)))
    lines = compute_lines_from_edge_candidate_clusters(
        img_edges.copy(),
        edge_candidate_clusters,
        mask_plane,
        mask_extract_contour,
        mask_number,
        output_directory
    )
    return lines


def split_edge_candidate_clusters(img, mask, keep_edges):
    from refiner.image_processing.draw import draw_corners_on_image
    from refiner.clustering.dbscan import dbscan_with_masked_image
    if keep_edges is None:
        return keep_edges
    masked_image = get_grayscaled_image(img.copy()) / 255
    masked_image[mask == 0] = 0
    mask_size = int(np.sum(masked_image))
    remove_keys = []
    new_edges = []
    for key, val in keep_edges.items():
        edge_size = get_n_unique_rows(val)
        if edge_size / mask_size > 0.05:
            logger.debug(
                "Splitting edge since it is big: {}% of total mask".format(round(100 * val.shape[0] / mask_size, 0)))
            img_keep_edges = draw_corners_on_image(val, get_colored_image(np.zeros_like(img)))
            corners = get_harris_corners(img_keep_edges)
            img_keep_edges = draw_corners_on_image(corners.tolist(), img_keep_edges, tuple([0, 0, 0]), radius=25)
            clustered_edges = dbscan_with_masked_image(get_grayscaled_image(img_keep_edges), eps=cfg.clustering_eps,
                                                       min_samples=cfg.clustering_min_sample)
            if len(clustered_edges) > 1:
                clustered_edges = filter_edges(masked_image * 255, clustered_edges)
                new_edges.append(clustered_edges)
                remove_keys.append(key)

    for key in remove_keys:
        del keep_edges[key]
    for i, edge in enumerate(new_edges):
        for key, val in edge.items():
            keep_edges['split_{}_{}'.format(i, key)] = val
    return keep_edges

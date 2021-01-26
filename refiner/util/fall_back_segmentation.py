import cv2
import numpy as np
import logging

from refiner.image_processing.evaluation_metrics import compute_iou_of_masks

logger = logging.getLogger(__name__)


def apply_fall_back_logic(refined_masks, fallback_masks, threshold=0.75):
    for i in range(len(refined_masks)):
        refined_mask = refined_masks[i]
        fallback_mask = fallback_masks[i]
        if compute_iou_of_masks(refined_mask, fallback_mask) < threshold:
            refined_masks[i] = fallback_masks[i]
    return refined_masks


def find_matching_nested_list(l, val):
    for sub_list in l:
        if val == sub_list[0]:
            return l.index(sub_list)
    return None


def compute_convex_hull_for_mask(mask):
    convex_hull = None
    indices = np.nonzero(mask)
    indices = np.array(indices).transpose()
    if indices.size > 0:
        convex_hull = cv2.convexHull(indices).reshape(-1, 2)
    return convex_hull


def compute_reduced_convex_hull_for_mask(mask, number_points):
    approx_hull = compute_convex_hull_for_mask(mask)
    fac = 0.005
    while approx_hull.shape[0] > number_points:  # reducing complexity of convex hull
        perimeter = cv2.arcLength(approx_hull, True)
        approx_hull = cv2.approxPolyDP(approx_hull, fac * perimeter, True)
        fac += 0.005
        if approx_hull.shape[0] < number_points:
            logger.debug("Approximated shape with only {} points".format(approx_hull.shape[0]))
    return approx_hull.reshape(-1, 2)

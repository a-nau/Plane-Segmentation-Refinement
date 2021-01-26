import cv2
import numpy as np

from refiner.image_processing.draw import draw_convex_hull_around_points_no_mask, draw_convex_hull_around_points
from utils.image_modification import get_colored_image, scale_image


def compute_iou_of_masks(mask0, mask1):
    if np.sum(mask0) == 0 or np.sum(mask1) == 0:
        return 0
    mask0 = (mask0 / np.max(mask0)).astype(np.uint8)
    mask1 = (mask1 / np.max(mask1)).astype(np.uint8)
    intersection = cv2.bitwise_and(mask0, mask0, mask=mask1)
    union = np.zeros_like(mask0)
    union[mask0 + mask1 > 0] = 1
    iou = np.sum(intersection) / np.sum(union)
    return iou


def compute_iou_of_mask_with_gt(refined_masks, gt_masks, draw=False):
    from refiner.image_processing.draw import draw_masks_on_image
    from utils.image_modification import scale_image
    ious = []
    for refined_mask in refined_masks:
        tmp_ious = []
        for gt_mask in gt_masks:
            iou = compute_iou_of_masks(refined_mask, gt_mask)
            if draw:
                img = cv2.cvtColor((gt_mask.copy().astype(np.uint8)), cv2.COLOR_GRAY2RGB)
                img = draw_masks_on_image(img, [refined_mask])
                cv2.imshow("", scale_image(np.hstack((gt_mask, refined_mask)), 0.6))
                cv2.waitKey(0)
            tmp_ious.append(iou)
        best_idx = np.argmax(tmp_ious)
        ious.append([best_idx, tmp_ious[best_idx]])
    return ious


def check_convexity_and_mask_not_empty(mask):
    if np.sum(mask) == 0:
        return False
    # check if mask is convex
    contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask_convex = draw_convex_hull_around_points_no_mask(contours[1][0], mask.shape)
    iou = np.sum(mask / np.max(mask)) / np.sum(mask_convex / np.max(mask_convex))
    if iou < 0.70:
        return False
    return True


def compare_iou_of_points_with_mask(points, mask, color, draw=False):
    line_mask = np.zeros_like(mask, dtype=np.uint8)
    points = np.array(points, dtype=np.float32)
    line_mask = draw_convex_hull_around_points(points, line_mask.copy(), color)
    union = cv2.bitwise_and(mask, mask, mask=line_mask)
    sum_line_mask = np.sum(line_mask / max(1, np.max(line_mask)))
    sum_union = np.sum(union / max(1, np.max(union)))
    if draw:
        img = get_colored_image(np.copy(mask))
        color1 = tuple([255, 0, 0])
        color2 = tuple([0, 255, 0])
        img = cv2.line(img, tuple(points[0]), tuple(points[1]), color1, 5)
        img = cv2.line(img, tuple(points[1]), tuple(points[2]), color2, 5)
        img = cv2.line(img, tuple(points[2]), tuple(points[3]), color2, 5)
        img = cv2.line(img, tuple(points[3]), tuple(points[0]), color2, 5)
        cv2.imshow("test", scale_image(img, 0.5))
        cv2.waitKey(0)
    if sum_line_mask == 0:
        return None
    iou = sum_union / sum_line_mask
    return iou

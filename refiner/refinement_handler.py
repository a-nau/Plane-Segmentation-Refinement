import logging
import os
import time

import cv2
import numpy as np

from refiner.clustering.edge_detection import compute_widened_contour_line_around_mask
from refiner.clustering.handler import get_lines_from_clustering
from refiner.combined.combined_appraoch import cluster_lines_together_and_choose_best
from refiner.hough_transform.handler import get_lines_from_hough_transformation
from refiner.hough_transform.hough_transformation import find_line_end_points_detailed
from refiner.image_processing.draw import draw_masks_on_image, draw_all_lines_on_image, draw_corners_on_image, \
    draw_convex_hull_around_points_no_mask
from refiner.image_processing.evaluation_metrics import compute_iou_of_mask_with_gt
from refiner.image_processing.visualize_results import visualize_results, visualize_summary
from refiner.models.image_data import ImageData
from refiner.util.fall_back_segmentation import apply_fall_back_logic
from utils.directories import clean_directory
from utils.image_modification import get_grayscaled_image, get_colored_image

from config import GlobalConfig as GlobalConfig
cfg = GlobalConfig.get_config()
logger = logging.getLogger(__name__)


def start_refinement_procedure(data: ImageData):
    # Compute refinement
    refined_masks = compute_refined_masks(data)
    refined_masks = apply_fall_back_logic(refined_masks, data.fallback_masks)

    # Accuracy analysis
    result_dict_planercnn = run_accuracy_analysis(data, data.planercnn_masks, 'planercnn')
    result_dict_fallback = run_accuracy_analysis(data, data.fallback_masks, 'fallback')
    result_dict_refined = run_accuracy_analysis(data, refined_masks, 'refined')
    result_dict = {**result_dict_planercnn, **result_dict_fallback, **result_dict_refined}
    logger.debug('Loaded planes')
    return result_dict


def compute_refined_masks(data: ImageData):
    dir_results = data.output_directory
    if not cfg.visualization_dict['no_result_folder']:
        clean_directory(dir_results)
    refined_masks = []
    all_lines = dict()

    if cfg.visualization_dict['edges']:
        cv2.imwrite(os.path.join(dir_results, "edges.jpg"), data.edge_img)
    if cfg.visualization_dict['initial_mask']:
        image_masked = draw_masks_on_image(data.edge_img, data.planercnn_masks)
        cv2.imwrite(os.path.join(dir_results, "masked_image.jpg"), image_masked)
    for i, mask in enumerate(data.planercnn_masks):
        logger.info(f"{'#' * 35} Running on plane {i} {'#' * 35}")
        mask, lines = compute_refined_mask_and_lines(data.img, data.edge_img, mask, dir_results, i)
        if mask is not None:
            refined_masks.append(mask)
        if lines is not None:
            all_lines[i] = lines
    if cfg.visualization_dict['refined_mask']:
        image_refined = draw_masks_on_image(get_colored_image(data.edge_img.copy()), refined_masks)
        cv2.imwrite(os.path.join(dir_results, "refined_masked_image.jpg"), image_refined)
    if cfg.visualization_dict['lines']:
        all_lines_image = draw_all_lines_on_image(data.img, all_lines)
        cv2.imwrite(os.path.join(dir_results,
                                 "image_all_lines_{}.png".format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))),
                    all_lines_image)
    return refined_masks


def compute_refined_mask_and_lines(img_org, img_edges, mask, dir_results, i):
    mask_extract_contour = compute_widened_contour_line_around_mask(mask, ksize=cfg.mask_size)
    logger.info("Computing lines with Hough Transformation")
    # Hough transform
    lines_hough = get_lines_from_hough_transformation(
        img_edges,
        mask_extract_contour,
        cluster_lines=True,
        output_directory=os.path.join(dir_results, cfg.dir_result_details)
    )
    # DBSCAN clustering
    logger.info("Computing lines with clustering")
    lines_clustering = get_lines_from_clustering(
        img_edges,
        mask_extract_contour,
        mask,
        i,
        ksize=cfg.mask_size,
        output_directory=os.path.join(dir_results, cfg.dir_result_details)
    )
    lines = lines_clustering + lines_hough
    if len(lines) == 0:
        # use planercnn mask as fallback
        return mask, None

    # Find best lines
    logger.info(f"Find best lines in {len(lines)} detected lines")
    lines = cluster_lines_together_and_choose_best(lines, get_grayscaled_image(img_edges.copy()))
    lines = find_line_end_points_detailed(get_grayscaled_image(img_edges.copy()), lines)
    points = np.array(sum(lines, [])).reshape(-1, 2)
    if lines is not None and len(lines) > 0:
        logger.info(f"Refining mask along the best {points.shape[0]} points")
        refined_mask = adjust_mask_with_points(img_org, points, mask)
        if refined_mask is not None:
            save_image_with_refined_mask(img_edges, refined_mask, points, i, dir_results)
            return refined_mask, lines
    return None, None


def save_image_with_refined_mask(img_edges, mask, points, i, dir_results):
    image_refined_curr = draw_masks_on_image(get_colored_image(img_edges.copy()), [mask])
    image_refined_curr = draw_corners_on_image(points, image_refined_curr, radius=4)
    if cfg.visualization_dict['refined_mask']:
        cv2.imwrite(os.path.join(dir_results, "mask_{}_refined.jpg".format(i)), image_refined_curr)


def run_accuracy_analysis(data: ImageData, masks, string_identification):
    ious_refined_masks = compute_iou_of_mask_with_gt(masks, data.gt_masks)
    ious_only_refined_masks = find_masks_best_matching_each_gt_mask(ious_refined_masks, len(data.gt_masks))
    result_dict = {'ious_{}_masks_{}'.format(string_identification, i): ious_only_refined_masks[i] for i in
                   range(len(data.gt_masks))}
    result_dict['ious_{}_masks_average'.format(string_identification)] = np.mean(ious_only_refined_masks)
    if cfg.visualization_dict['summary']:
        visualize_results(data.img, data.gt_masks, masks, ious_refined_masks, data.output_directory,
                          data.id + '_' + string_identification)
    visualize_summary(data.img, data.gt_masks, masks, ious_refined_masks,
                      data.id + '_' + string_identification)
    return result_dict


def find_masks_best_matching_each_gt_mask(masks, n_gt_masks):
    ious = []
    for i in range(n_gt_masks):
        candidates = [x[1] for x in masks if x[0] == i]
        if len(candidates) > 0:
            idx = np.argmax(candidates)
            ious.append(candidates[idx])
        else:
            ious.append(0)
    return ious


def adjust_mask_with_points(image, points, mask):
    new_mask = draw_convex_hull_around_points_no_mask(points, mask.shape)
    return new_mask

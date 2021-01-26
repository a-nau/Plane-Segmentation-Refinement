import logging
import os

import cv2
import numpy as np

from config import GlobalConfig as GlobalConfig
from refiner.image_processing.draw import draw_masks_on_image

cfg = GlobalConfig.get_config()
logger = logging.getLogger(__name__)


def visualize_results(img, gt_masks, masks, ious, output_directory, string_identification="", draw=False):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    for i, (iou, mask) in enumerate(zip(ious, masks)):
        if iou[1] > 0.5:
            img_mask = draw_ground_truth_and_estimation_on_image(gt_masks[iou[0]], img.copy(), mask, cfg.color_pallet[i])
            cv2.putText(img_mask,
                        "Mask with IoU {0:.2f}%".format(iou[1] * 100),
                        tuple(int(0.1 * mask.shape[i]) for i in range(len(mask.shape))[::-1]),  # position
                        cv2.FONT_HERSHEY_SIMPLEX,  # font family
                        1,  # font size
                        color=(209, 80, 0, 255),  # font color
                        thickness=2)  # font stroke

            cv2.imwrite(os.path.join(output_directory,
                                     "{}_{}.png".format(string_identification, i)), img_mask)
            if draw:
                cv2.imshow("", img_mask)
                cv2.waitKey(0)


def visualize_summary(img, gt_masks, masks, ious, string_identification="", draw=False):
    if not os.path.exists(cfg.path_summary):
        os.mkdir(cfg.path_summary)
    img_mask = img.copy()
    for i, (iou, mask) in enumerate(zip(ious, masks)):
        if iou[1] > 0.4:
            img_mask = draw_ground_truth_and_estimation_on_image(gt_masks[iou[0]], img_mask, mask, cfg.color_pallet[i])
    cv2.imwrite(os.path.join(cfg.path_summary, "{}.png".format(string_identification)), img_mask)
    if draw:
        cv2.imshow("", img_mask)
        cv2.waitKey(0)


def draw_ground_truth_and_estimation_on_image(gt_mask, img_mask, mask, color):
    contours = cv2.findContours(gt_mask.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_mask = draw_masks_on_image(img_mask, [mask.astype(np.uint8)], color)  # draw estimation
    img_mask = cv2.drawContours(img_mask, contours[0], 0, tuple([255, 0, 0]), thickness=4)  # draw ground truth
    return img_mask

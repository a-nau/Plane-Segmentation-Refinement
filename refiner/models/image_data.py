import logging
import os

import cv2
import numpy as np

from config import GlobalConfig as GlobalConfig
from refiner.image_processing.draw import draw_convex_hull_around_points_no_mask, draw_contour_around_points_no_mask, \
    enlarge_contours
from refiner.image_processing.edge_detection import canny_edge_detection
from refiner.image_processing.evaluation_metrics import compute_iou_of_masks
from refiner.util.fall_back_segmentation import compute_reduced_convex_hull_for_mask
from utils.image_modification import get_colored_image

cfg = GlobalConfig.get_config()
logger = logging.getLogger(__name__)


class ImageData:
    def __init__(self, image_number, dir_data):
        self.number = image_number
        self.data_directory = dir_data
        self.id = os.path.basename(dir_data)
        self.output_directory = os.path.join(cfg.path_output, f"{self.id}_{cfg.dir_results}")
        self.output_details_directory = os.path.join(self.output_directory, cfg.dir_result_details)

        self.planes = []
        self.planercnn_masks = []
        self.fallback_masks = []
        self.refined_masks = []
        self.img = None
        self.edge_img = None
        self.ious = []

        # Load data from directory
        self.plane_masks_file = os.path.join(self.data_directory, cfg.file_name_plane_masks(self.number))
        self.original_image_file = os.path.join(self.data_directory, cfg.file_name_input_image)
        self.planercnn_image_file_path = os.path.join(self.data_directory, cfg.file_name_planercnn_image(self.number))

    def load_data(self):
        # Load mask data
        self.load_plane_data()
        self.load_ground_truth_masks()
        # Load image data
        self.img = ImageData.load_image(self.original_image_file)
        self.get_edge_img()

    def load_plane_data(self):
        plane_masks = np.load(self.plane_masks_file)
        self.planercnn_masks = []
        self.fallback_masks = []
        for i in range(plane_masks.shape[0]):
            mask = plane_masks[i, :, :].transpose().astype(np.int16)
            mask = cv2.resize(mask, cfg.image_size[::-1])
            fallback_mask = self.compute_fallback_mask(mask).astype(np.uint8)
            iou = compute_iou_of_masks(mask.transpose(), fallback_mask)
            if iou > 0.75 and np.sum(mask) > 0:
                self.fallback_masks.append(fallback_mask)
                self.planercnn_masks.append(mask)
        self.planercnn_masks = [(mask * 255).transpose().astype(np.uint8) for mask in self.planercnn_masks]

    def load_ground_truth_masks(self):
        try:
            self.gt_masks = self.load_ground_truth_masks_from_directory(
                self.data_directory, "via_region_data",
                self.planercnn_masks[0].shape
            )
        except:
            self.gt_masks = self.planercnn_masks
            logger.warning("Did not find ground truth plane segmentation; using planercnn ones instead")
        self.gt_masks = [mask.astype(np.uint8) for mask in self.gt_masks]

    def compute_fallback_mask(self, mask):
        approx_convex_hull = compute_reduced_convex_hull_for_mask(mask, 20)
        fallback_mask = draw_convex_hull_around_points_no_mask(approx_convex_hull,
                                                               mask.transpose().shape)
        return fallback_mask

    def get_edge_img(self):
        if cfg.edge_detection_type in ['canny', 'canny_adap']:
            self.edge_img_file = os.path.join(self.data_directory, "image.png")
            image = self.load_image(self.edge_img_file)
            canny_apative = True if cfg.edge_detection_type == 'canny_adap' else False
            img_edges = get_colored_image(canny_edge_detection(image, 30, 70, canny_apative))
        else:
            self.edge_img_file = os.path.join(self.data_directory,
                                              "image_contour{}.png".format(cfg.edge_detection_type))
            image = self.load_image(self.edge_img_file)
            img_edges = 255 - image
            threshold = 225
            img_edges[img_edges < threshold] = 0
            img_edges[img_edges > threshold] = 255
        self.edge_img = enlarge_contours(img_edges, ksize=1)

    @staticmethod
    def load_ground_truth_masks_from_directory(directory, identification_string, mask_shape):
        import json
        masks = []
        files = [f for f in os.listdir(directory) if
                 (os.path.isfile(os.path.join(directory, f)) and identification_string in f)]
        masks_json = []
        for file in files:
            with open(os.path.join(directory, file)) as f:
                masks_json.append(json.load(f))
        for mask_json in masks_json:
            for _, val in mask_json.items():
                rectangle = val['regions']
                for _, val_region in rectangle.items():
                    points = np.array([val_region['shape_attributes']['all_points_x'],
                                       val_region['shape_attributes']['all_points_y']]).transpose()
                    mask = draw_contour_around_points_no_mask(points, mask_shape)
                    masks.append(mask)
        return masks

    @staticmethod
    def load_image(path):
        return cv2.resize(cv2.imread(path), cfg.image_size)

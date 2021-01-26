import logging
import random
import warnings

import cv2
import numpy as np

from config import GlobalConfig as GlobalConfig
from utils.image_modification import get_grayscaled_image, get_colored_image

cfg = GlobalConfig.get_config()
logger = logging.getLogger(__name__)


def draw_convex_hull_around_points(points, mask, color=255):
    convex_hull = cv2.convexHull(points.astype(np.float32))  # need float input
    convex_hull = convex_hull.astype(np.int32).reshape(-1, 2)  # fill needs int input
    mask = cv2.fillConvexPoly(get_colored_image(mask), convex_hull, tuple([color, color, color]))
    mask = get_grayscaled_image(mask)
    return mask


def draw_convex_hull_around_points_no_mask(points, mask_shape, color=255):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mask = np.zeros(mask_shape)
        if isinstance(points, list):
            if len(points) == 0:
                return np.zeros_like(mask_shape)
            else:
                points = np.array(points)
        convex_hull = cv2.convexHull(points.astype(np.float32))  # need float input
        convex_hull = convex_hull.astype(np.int32).reshape(-1, 2)  # fill needs int input
        mask = cv2.fillConvexPoly(get_colored_image(mask), convex_hull, tuple([color, color, color]))
        mask = get_grayscaled_image(mask)
        return mask


def draw_contour_around_points_no_mask(points, mask_shape, color=255):
    mask = np.zeros(mask_shape)
    if isinstance(points, list):
        if len(points) == 0:
            return np.zeros_like(mask_shape)
        else:
            points = np.array(points)
    mask = cv2.drawContours(mask, [points], -1, tuple([color]), thickness=-1)
    mask = get_grayscaled_image(mask)
    return mask


def draw_corners_on_image(corners, img, color=None, radius=2):
    color = random.choice(cfg.color_pallet) if color is None else color
    for corner in corners:
        if isinstance(corner, (list, np.ndarray)):
            corner = tuple((int(corner[0]), int(corner[1])))
        img = cv2.circle(img, corner, radius=radius, color=color, thickness=-1)
    return img


def draw_lines_on_image(image, lines, color=None, thickness=2):
    color = random.choice(cfg.color_pallet) if color is None else color
    new_image = image.copy()
    for line in lines:
        new_image = cv2.line(new_image,
                             tuple(int(i) for i in line[0]),
                             tuple(int(i) for i in line[1]),
                             color,
                             thickness)
    return new_image


def draw_lines_rho_theta_on_image(img, lines, color=None, thickness=2):
    color = random.choice(cfg.color_pallet) if color is None else color
    line_image = np.copy(img)

    for line in lines:
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
        cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

    return line_image


def draw_masks_on_image(image, masks, color=None):
    masked_image = image.copy()
    for i, mask in enumerate(masks):
        curr_color = cfg.color_pallet[i] if color is None else color
        mask_raw = np.divide(get_colored_image(mask.astype(np.float).copy()), 255)
        alpha = np.zeros_like(image).astype(np.uint8)
        alpha[:, :, 0] = mask_raw
        alpha[:, :, 1] = mask_raw
        alpha[:, :, 2] = mask_raw
        background = np.zeros_like(image)  # use colored image for colored masks
        background[:] = curr_color
        masked_area = cv2.multiply(alpha, background)
        masked_image = cv2.add(masked_image, masked_area)
    return masked_image


def draw_all_lines_on_image(image, lines_dict):
    lines_image = image.copy()
    for id, lines in lines_dict.items():
        color = cfg.color_pallet[id]
        lines_image = draw_lines_on_image(lines_image, lines, color, thickness=3)
    return lines_image


def enlarge_contours(image, ksize):
    image = get_grayscaled_image(image)
    mask = cv2.GaussianBlur(image, (ksize, ksize), 0)  # enlarge mask
    image[mask > 0] = 255
    return get_colored_image(image)

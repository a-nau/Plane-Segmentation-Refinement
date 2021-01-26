import cv2
import numpy as np

from config import GlobalConfig as GlobalConfig
from utils.image_modification import get_grayscaled_image

cfg = GlobalConfig.get_config()


def fast_corner_detection(img):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html
    fast = cv2.FastFeatureDetector_create()
    img = get_grayscaled_image(img)

    # find and draw the keypoints
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, color=(255, 0, 0), outImage=img)
    return img2, [tuple((int(keypoint.pt[0]), int(keypoint.pt[1]))) for keypoint in kp]


def harris_corner_detection(img):
    img_gray = get_grayscaled_image(img)
    corners = cv2.cornerHarris(img_gray, 2, 3, 0.04)
    corners2 = cv2.dilate(corners, None, iterations=3)
    img[corners2 > 0.01 * corners2.max()] = [255, 0, 0]
    return img


def get_harris_corners(img):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, cfg.kernel_harris_corner_detector, 5, 0.1)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    return corners

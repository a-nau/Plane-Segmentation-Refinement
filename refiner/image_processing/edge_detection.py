import cv2
import numpy as np

from utils.image_modification import get_grayscaled_image


def canny_edge_detection(img, threshold1=100, threshold2=200, adaptive=True):
    imgray = get_grayscaled_image(img)
    if adaptive:
        threshold2, _ = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        threshold1 = 0.4 * threshold2
    edges = cv2.Canny(imgray, threshold1, threshold2)
    return edges


def canny_edge_detection_with_automatic_threshold(image, sigma=0.33):
    # https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def get_contours(img):
    try:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        imgray = img
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return im

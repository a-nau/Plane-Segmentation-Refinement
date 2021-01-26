import os

from utils.image_modification import get_grayscaled_image


def get_lines_from_hough_transformation(image, mask, cluster_lines, output_directory=None):
    from refiner.hough_transform.hough_transformation import detect_lines
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    prob = False
    img_masked = get_grayscaled_image(image.copy())
    img_masked[mask == 0] = 0
    lines = detect_lines(img_masked, probabilistic=prob, cluster_lines=cluster_lines,
                         output_directory=output_directory)
    return lines

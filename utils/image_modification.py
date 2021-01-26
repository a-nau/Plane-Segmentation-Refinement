import cv2


def scale_image(img, scale=0.5):
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img = cv2.resize(img, new_size)
    return img


def get_width_cropped_image(image, roi):
    img_cropped = image[:, roi[0]:(roi[0] + roi[2])]
    return img_cropped


def load_scale_and_resave_picture(src_path, dst_path, scale):
    image_cv = cv2.imread(src_path)
    scale = float(scale)
    cv2.imwrite(dst_path, scale_image(image_cv, scale))


def get_grayscaled_image(img):
    if len(img.shape) == 3:
        try:
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            imgray = img
    else:
        imgray = img
    return imgray


def get_colored_image(img):
    if len(img.shape) == 2:
        try:
            img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        except:
            img_colored = img
    else:
        img_colored = img
    return img_colored

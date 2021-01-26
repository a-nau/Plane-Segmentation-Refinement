import os
import random


def check_if_file_is_image(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))


def find_all_image_files_in_directory(directory):
    img_list = [f for f in os.listdir(directory) if
                (os.path.isfile(os.path.join(directory, f)) and check_if_file_is_image(f))]
    img_list.sort()
    return img_list


def select_subset_from_list(list_data, subset_size):
    if subset_size is None:
        subset_size = len(list_data)
    else:
        subset_size = min(len(list_data), subset_size)
    list_data = random.sample(list_data, subset_size)
    return list_data

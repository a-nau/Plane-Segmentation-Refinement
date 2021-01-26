import os
import shutil
import time


def clean_directory(dir):
    for directory in [dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        try:
            os.makedirs(directory)
        except PermissionError:
            print("waiting to create folder")
            time.sleep(4)
            os.makedirs(directory)


def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

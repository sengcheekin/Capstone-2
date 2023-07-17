# Code taken and adapted from https://github.com/tranleanh/haze-synthesize/tree/master

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import glob

def get_file_name(path):
    basename = os.path.basename(path)
    onlyname = os.path.splitext(basename)[0]
    return onlyname


def gen_haze(img, depth_img):
    # To make it so darker areas = further distance
    max_depth_value = np.max(depth_img)
    depth_img = max_depth_value - depth_img

    depth_img_3c = np.zeros_like(img)
    depth_img_3c[:, :, 0] = depth_img
    depth_img_3c[:, :, 1] = depth_img
    depth_img_3c[:, :, 2] = depth_img

    beta = random.uniform(2.4, 3.0)
    norm_depth_img = depth_img_3c / 255
    trans = np.exp(-norm_depth_img * beta)

    A = 255
    hazy = img * trans + A * (1 - trans)
    hazy = np.array(hazy, dtype=np.uint8)

    return hazy


img_path = "D:\Documents\Semester 9\Capstone 2\Code\Capstone-2\datasets\highres/clean"
depth_path = "D:\Documents\Semester 9\Capstone 2\Code\Capstone-2\datasets\highres/depth_images"
ext = "jpg"

# Search the folder for images
paths = glob.glob(os.path.join(img_path, "*.{}".format(ext)))
output_dir = "D:\Documents\Semester 9\Capstone 2\Code\Capstone-2\datasets\highres/hazy"

if __name__ == "__main__":
    for path in paths:
        fname = get_file_name(path)
        img = plt.imread(path)
        depth_img_path = os.path.join(depth_path, "{}_depth.jpg".format(fname))
        depth_img = plt.imread(depth_img_path)

        hazy = gen_haze(img, depth_img)
        output_path = os.path.join(output_dir, "{}_hazy.jpg".format(get_file_name(path)))
        plt.imsave(output_path, hazy)
        print("Saved {}".format(output_path))

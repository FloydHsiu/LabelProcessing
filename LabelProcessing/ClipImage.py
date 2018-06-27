import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
from os.path import isdir, isfile, join
from os import listdir, mkdir
import json
import tqdm


def clip_shiftx(src_image, annotations, stitch_size, clip_size, stride=2):
    # Input:
    #   src_image : np_2d_array(height, width)
    #   bbox : np_3d_array(bbox_num, bbox_size, 2)
    #   stitch_size : int --> src_image is stitched by how many images
    #   clip_size : int --> clip_image is stitched by how many images
    #   stride : int --> -1 as auto stride, decide by num
    # Output:
    #   clip_images : list(np_2d_array)
    #   clip_annotations_list : list(list([x,y], [x,y],....))
    clip_width = clip_size * src_image.shape[1] / stitch_size
    stride_width = stride * src_image.shape[1] / stitch_size
    clip_nums = int(np.floor((stitch_size - clip_size) / stride))
    clip_images = []
    clip_annotations_list = []

    for i in range(0, clip_nums):
        # clip image
        clip_start = int(i * stride_width)
        clip_end = int(clip_width + clip_start)
        clip_image = src_image[:, clip_start: clip_end]
        clip_annotations = []
        # clip bbox
        for a in annotations:
            a = a.copy()
            isAllClear = True
            new_box = []
            for point in a['bbox']:
                x = point[0]
                if x >= clip_start and x < clip_end:
                    isAllClear = True
                    new_box.append([x-clip_start, point[1]])
                else:
                    isAllClear = False
                    break
            if isAllClear:
                a['bbox'] = new_box
                clip_annotations.append(a)
        clip_images.append(clip_image)
        clip_annotations_list.append(clip_annotations)
    return clip_images, clip_annotations_list


def load_images_and_labels(image_dir, label_dir):
    # Input:
    #   image_dir : string --> images directory
    #   label_dir : string --> label directory
    # Output:
    #   images : list(np_2d_array)
    #   labels : list(dict)
    #   names : list(string) --> filenames
    images = []
    labels = []
    names = []
    if isdir(image_dir):
        for fn in tqdm.tqdm(listdir(image_dir), desc='Load Images, Labels'):
            fn_path = join(image_dir, fn)
            fn_substr = fn.split('.')
            fn_label_path = join(label_dir, f'{fn_substr[0]}.json')
            if isfile(fn_path) and isfile(fn_label_path) and fn_substr[-1] == 'jpg':
                images.append(cv.imread(fn_path, cv.IMREAD_GRAYSCALE))
                with open(fn_label_path, 'r') as f:
                    labels.append(json.load(f))
                names.append(fn_substr[0])
        print(f'Success Load {len(images)} images')
    return images, labels, names

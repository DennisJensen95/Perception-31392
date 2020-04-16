from lib.fastRCNNPretrained import *
from lib.prepareData import *
import glob

test_1 = False
test_2 = False
test_3 = True

if test_1:

    images = sorted(glob.glob('./data/stereo_conveyor_without_occlusions/left/*.png'))

    image = images[1330]
    debug = False
    img = cv2.imread(image)
    if debug:
        cv2.imshow('test', img)
        cv2.waitKey(500)

    object_detection_api(image, 50, threshold=0.7)

    images = getImages('data/stereo_conveyor_without_occlusions/left/')

    box_1 = [100, 250, 'box']
    box_2 = [330, 450, 'box']
    book_1 = [500, 630, 'book']
    book_2 = [660, 830, 'book']
    book_3 = [870, 1030, 'book']
    cup_1 = [1070, 1220, 'cup']
    cup_2 = [1260, 1460, 'cup']

    intervals = [book_1, book_2, book_3, cup_1, cup_2]
    y = np.empty(0)
    x = np.empty(0)
    for i in range(len(intervals)):
        object_images = np.asarray(images[intervals[i][0]:intervals[i][1]])
        frames = intervals[i][1] - intervals[i][0]
        y_objects = np.full(frames, intervals[i][2])
        y = np.append(y, y_objects)
        x = np.append(x, object_images)


    for image in x:
        object_detection_api(image, showtime=10, threshold=0.8)

if test_2:
    import numpy as np
    import time
    import sys
    import os
    import random
    from skimage import io
    import pandas as pd
    from matplotlib import pyplot as plt
    from shutil import copyfile

    import cv2
    import tensorflow as tf

    base_path = 'rcnn-test-data/'
    images_boxable_fname = 'train-images-boxable.csv'
    annotations_bbox_fname = 'oidv6-train-annotations-bbox.csv'
    class_descriptions_fname = 'class-descriptions-boxable.csv'
    images_boxable = pd.read_csv(os.path.join(base_path, images_boxable_fname))
    print(images_boxable.head())
    annotations_bbox = pd.read_csv(os.path.join(base_path, annotations_bbox_fname))
    print(annotations_bbox.head())
    class_descriptions = pd.read_csv(os.path.join(base_path, class_descriptions_fname))
    print(class_descriptions.head())

    print('length of the images_boxable: %d' % (len(images_boxable)))
    print('First image in images_boxable👇')
    img_name = images_boxable['image_name'][0]
    img_url = images_boxable['image_url'][0]
    print('\t image_name: %s' % (img_name))
    print('\t img_url: %s' % (img_url))
    print('')
    print('length of the annotations_bbox: %d' % (len(annotations_bbox)))
    print('The number of bounding boxes are larger than number of images.')
    print('')
    print('length of the class_descriptions: %d' % (len(class_descriptions) - 1))
    img = io.imread(img_url)

if test_3:
    """ Test trained model """


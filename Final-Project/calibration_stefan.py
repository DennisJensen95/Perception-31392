# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:45:49 2020

@author: Stefan Carius Larsen
"""

import os
import cv2
import matplotlib.pyplot
import numpy as np
import glob

os.chdir(r"C:\Users\Stefan Carius Larsen\Google Drive\Danmarks Tekniske Universitet\Kandidat - Elektroteknologi\1. Semester\31392_Perception_for_Autonomous_Systems\Perception-31392\Final-Project")

images_left = sorted(glob.glob("data/Stereo_calibration_images/left*.png"))
images_right = sorted(glob.glob("data/Stereo_calibration_images/right*.png"))
images_left_conveyor = sorted(glob.glob("data/Small_dataset_no_occlusion/left/*.png"))
images_right_conveyor = sorted(glob.glob("data/Small_dataset_no_occlusion/right/*.png"))

images_conveyor = np.asarray([images_left_conveyor, images_right_conveyor]).T
images = np.asarray([images_left, images_right]).T

nb_vertical = 6
nb_horizontal = 9

# Sub pixels criteria
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

# Prepare data objects

objp = np.zeros((nb_horizontal * nb_vertical, 3), np.float32)
objp[:, :2] = np.mgrid[0:nb_vertical, 0:nb_horizontal].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints_left = []  # 2d points in image plane.
imgpoints_right = []  # 2d points in image plane.
img_shape = None

for images in images:
    # Read images
    left_img = cv2.imread(images[0])
    right_img = cv2.imread(images[1])

    # Greyscale
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_left, (nb_vertical, nb_horizontal), None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_right, (nb_vertical, nb_horizontal), None)

    # Placeholder for the number of images
    objpoints.append(objp)
    if ret_l:
        corners_l = cv2.cornerSubPix(gray_left, corners_l, (11, 11), (-1, -1), criteria)

        imgpoints_left.append(corners_l)
        

    if ret_r:
        corners_r = cv2.cornerSubPix(gray_right, corners_r, (11, 11), (-1, -1), criteria)

        imgpoints_right.append(corners_r)



left_camera_calibration_data = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
right_camera_calibration_data = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

calibrated_cameras = True
objpoints = objpoints
imgpoints_l = imgpoints_left
imgpoints_r = imgpoints_right




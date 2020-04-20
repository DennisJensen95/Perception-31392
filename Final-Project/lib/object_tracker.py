import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from lib.calibration import Calibration

os.chdir("C:/Users/Stefan Larsen/Google Drev/Danmarks Tekniske Universitet/Kandidat - Elektroteknologi/1. Semester/31392_Perception_for_Autonomous_Systems/Perception-31392/Final-Project")

images_left = sorted(glob.glob("data/Stereo_calibration_images/left*.png"))
images_right = sorted(glob.glob("data/Stereo_calibration_images/right*.png"))
images_left_conveyor = sorted(glob.glob("data/Small_dataset_no_occlusion/left/*.png"))
images_right_conveyor = sorted(glob.glob("data/Small_dataset_no_occlusion/right/*.png"))

images_conveyor = np.asarray([images_left_conveyor, images_right_conveyor]).T
images = np.asarray([images_left, images_right]).T
nb_vertical = 6
nb_horizontal = 9
Cal = Calibration(images, nb_vertical=nb_vertical, nb_horizontal=nb_horizontal)
Cal.calibrateCamera(debug=False)
Cal.stereoCalibration()

images = images_conveyor[0]
Cal.remapImagesStereo(images, random=False, debug=False)

i = 0

while True:
    img_l = cv2.imread(images_conveyor[i][0])
    img_r = cv2.imread(images_conveyor[i][1])
    
    """-----------------------------------------"""
    """ RGB TO GRAY """
    """-----------------------------------------"""
    
    img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    
    """-----------------------------------------"""
    """ USING BLUR TO DETECT MOVING OBJECT """
    """-----------------------------------------"""
    
    img_l = cv2.medianBlur(img_l, 7)
    img_r = cv2.medianBlur(img_r, 7)
    
    
    
    
    
    
    
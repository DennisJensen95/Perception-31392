import glob
import numpy as np
from lib.calibration import Calibration
from lib.construct3D import *
import cv2

images_left = sorted(glob.glob("data/Stereo_calibration_images/left*.png"))
images_right = sorted(glob.glob("data/Stereo_calibration_images/right*.png"))
images_left_conveyor = sorted(glob.glob("data/stereo_conveyor_without_occlusions/left/*.png"))
images_right_conveyor = sorted(glob.glob("data/stereo_conveyor_without_occlusions/right/*.png"))
images_conveyor = np.asarray([images_left_conveyor, images_right_conveyor]).T
images = np.asarray([images_left, images_right]).T

calibrate = False
nb_vertical = 6
nb_horizontal = 9
Cal = Calibration(images, nb_vertical=nb_vertical, nb_horizontal=nb_horizontal)
if calibrate:
    Cal.calibrateCamera(debug=False)
    Cal.stereoCalibration()
    Cal.save_remapping_instance('RemappingData')
else:
    Cal.load_remapping_instance('RemappingData')

images = images_conveyor[0]

## Stereo Class
win_size = 5
min_disp = -1
max_disp = 63 #min_disp * 9
num_disp = max_disp - min_disp # Needs to be divisible by 16
#Create Block matching object.
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
 numDisparities = num_disp,
 blockSize = 5,
 uniquenessRatio = 5,
 speckleWindowSize = 5,
 speckleRange = 5,
 disp12MaxDiff = 1,
 P1 = 8*3*win_size**2,#8*3*win_size**2,
P2 =32*3*win_size**2) #32*3*win_size**2)

# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)

stop_iter = 350
for i, images in enumerate(images_conveyor):
    if stop_iter == i:
        break
    left_img, right_img = Cal.remapImagesStereo(images, random=False, debug=False)
    left_img = downsample_image(left_img, 0.5)
    right_img = downsample_image(right_img, 0.5)
    disp_img = stereo.compute(left_img, right_img)

    images = np.concatenate((left_img, right_img), axis=1)

    cv2.imshow('Images', disp_img)
    cv2.waitKey(10)




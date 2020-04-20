import glob
import numpy as np
from lib.calibration import Calibration
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
min_disp = 1
num_disp  = 10 * 16
block_size = 9
stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
stereo.setMinDisparity(min_disp)
stereo.setDisp12MaxDiff(200)
stereo.setUniquenessRatio(5)
stereo.setSpeckleRange(3)
stereo.setSpeckleWindowSize(30)

stop_iter = 30
for images in images_conveyor:
    left_img, right_img = Cal.remapImagesStereo(images, random=False, debug=False)
    disp_img = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

    images = np.concatenate((left_img, right_img), axis=1)

    cv2.imshow('Images', disp_img)
    cv2.waitKey(10)




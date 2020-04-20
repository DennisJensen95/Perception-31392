import glob
import numpy as np
from lib.calibration import Calibration
import os
import cv2
import matplotlib.pyplot as plt

#print("Current Working Directory " , os.getcwd())
os.chdir(r"C:\Users\Stefan Carius Larsen\Google Drive\Danmarks Tekniske Universitet\Kandidat - Elektroteknologi\1. Semester\31392_Perception_for_Autonomous_Systems\Perception-31392\Final-Project")

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
#Cal.remapImagesStereo(images, random=False, debug=True)
#Cal.remapImagesStereo(images, random=False, debug=False)


min_disp = 3
num_disp = 4 * 16
block_size = 19

for i in range(images_conveyor.shape[0]):
    images = images_conveyor[i]
    rect_L, rect_R = Cal.remapImagesStereo(images, random=False, debug=False)
    
    stereo = cv2.StereoSGBM_create(numDisparities=num_disp, blockSize=block_size)
    stereo.setMinDisparity(min_disp)
    stereo.setDisp12MaxDiff(1)
    stereo.setUniquenessRatio(1)
    stereo.setSpeckleRange(5)
    stereo.setSpeckleWindowSize(5)
    
    disp = stereo.compute(rect_L, rect_R).astype(np.float32) / 16.0
    plt.figure(figsize=(9,9))
    plt.imshow(disp, 'gray')
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #input("Press Enter to continue...")


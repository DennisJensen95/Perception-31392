import glob
import numpy as np
from lib.calibration import Calibration

images_left = sorted(glob.glob("data/Stereo_calibration_images/left*.png"))
images_right = sorted(glob.glob("data/Stereo_calibration_images/right*.png"))
images_left_conveyor = sorted(glob.glob("data/stereo_conveyor_without_occlusions/left/*.png"))
images_right_conveyor = sorted(glob.glob("data/stereo_conveyor_without_occlusions/right/*.png"))
images_conveyor = np.asarray([images_left_conveyor, images_right_conveyor]).T
images = np.asarray([images_left, images_right]).T
nb_vertical = 6
nb_horizontal = 9
Cal = Calibration(images, nb_vertical=nb_vertical, nb_horizontal=nb_horizontal)
Cal.calibrateCamera(debug=False)
Cal.stereoCalibration()

images = images_conveyor[0]
Cal.remapImagesStereo(images, random=False, debug=True)



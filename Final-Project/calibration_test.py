import glob
import numpy as np
from lib.calibration import Calibration

images_left = sorted(glob.glob("Stereo_calibration_images/left*.png"))
images_right = sorted(glob.glob("Stereo_calibration_images/right*.png"))
images = np.asarray([images_left, images_right]).T
nb_vertical = 6
nb_horizontal = 9
Cal = Calibration(images, nb_vertical=nb_vertical, nb_horizontal=nb_horizontal)
Cal.calibrateCamera(debug=False)
Cal.stereoCalibration()
Cal.remapImagesStereo(random=True, debug=True)

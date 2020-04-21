import glob
import numpy as np
from lib.calibration import Calibration
from lib.exportPC import export_pointcloud
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

# Stereo Class
min_disparity = 5  # 2
num_disparity = 10 * 16  # 160
block_size = 10  # 9
stereo = cv2.StereoSGBM_create(numDisparities=num_disparity, blockSize=block_size)
stereo.setMinDisparity(min_disparity)
stereo.setDisp12MaxDiff(200)  # 200
stereo.setUniquenessRatio(2)  # 5
stereo.setSpeckleRange(3)  # 3
stereo.setSpeckleWindowSize(30)  # 30

images_conveyor = images_conveyor[100:]  # start from first box with 80:

# stop_iter = 2
for images in images_conveyor:
    # if stop_iter == i:
    #    break
    left_img, right_img = Cal.remapImagesStereo(images, random=False, debug=False)
    # images = np.concatenate((left_img, right_img), axis=1)

    disparity_img = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    norm_disparity_img = cv2.normalize(disparity_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)

    cv2.imshow('Images', norm_disparity_img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# export_pointcloud(disparity_map=disparity_img, colors=left_img)
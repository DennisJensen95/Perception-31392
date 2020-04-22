import glob
import numpy as np
from lib.calibration import Calibration
from lib.construct3D import *
import cv2
from lib.sliderProgram import Slider

def main():
    images_left = sorted(glob.glob("data/Stereo_calibration_images/left*.png"))
    images_right = sorted(glob.glob("data/Stereo_calibration_images/right*.png"))
    images_left_conveyor = sorted(glob.glob("data/stereo_conveyor_without_occlusions/left/*.png"))
    images_right_conveyor = sorted(glob.glob("data/stereo_conveyor_without_occlusions/right/*.png"))
    images_conveyor = np.asarray([images_left_conveyor, images_right_conveyor]).T
    images = np.asarray([images_left, images_right]).T

    calibrate = False
    debug_disp_slider = False


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
    min_disparity = 1  # 1
    num_disparity = 75  # 160
    block_size = 9  # 5
    disp12MaxDiff = 130 # 200
    uniqRatio = 1 # 2
    speckleRange = 20 # 2
    speckleWindow = 125 # 30
    preFilter = 30 # 30
    p1 = 100
    p2 = 400
    mode = cv2.STEREO_SGBM_MODE_HH

    stereo = cv2.StereoSGBM_create(numDisparities=num_disparity,
                                   blockSize=block_size,
                                   minDisparity=min_disparity,
                                   disp12MaxDiff=disp12MaxDiff,
                                   uniquenessRatio=uniqRatio,
                                   speckleRange=speckleRange,
                                   speckleWindowSize=speckleWindow,
                                   preFilterCap=preFilter,
                                   mode=mode)

    images_conveyor = images_conveyor[0:-1]  # start from first box with 80:

    for images in images_conveyor:
        left_img, right_img = Cal.remapImagesStereo(images, random=False, debug=False)
        # images = np.concatenate((left_img, right_img), axis=1)
        left_img = downsample_image(left_img, 0.4)
        right_img = downsample_image(right_img, 0.4)

        if debug_disp_slider:
            slider = Slider(stereo, left_img, right_img)
            slider.create_slider()

        disparity_img = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
        norm_disparity_img = cv2.normalize(disparity_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                           dtype=cv2.CV_32F)

        cv2.imshow('Images', norm_disparity_img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # export_pointcloud(disparity_map=disparity_img, colors=left_img)


if __name__ == '__main__':
    main()
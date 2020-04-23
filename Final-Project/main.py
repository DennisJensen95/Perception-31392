import glob
import numpy as np
import imutils
# from skimage.metrics import structural_similarity as ssim
from lib.calibration import Calibration
from lib.construct3D import *
from lib.tracking import *
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
        # Cal.save_remapping_instance('ClassDataSaved')
        Cal.save_class('ClassDataSaved')
    else:
        # Cal.load_remapping_instance('ClassDataSaved')
        Cal.load_class('ClassDataSaved')

    # extract left/right images
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

    left_img, _ = Cal.remapImagesStereo(images_conveyor[0], random=False, debug=False)
    reference_img = cv2.GaussianBlur(left_img, (11, 11), 0)  # reference image to be used for tracking

    for images in images_conveyor[350:]:
        left_img, right_img = Cal.remapImagesStereo(images, random=False, debug=False)
        # images = np.concatenate((left_img, right_img), axis=1)
        # left_img = downsample_image(left_img, 0.5)
        # right_img = downsample_image(right_img, 0.5)
        #
        # if debug_disp_slider:
        #     slider = Slider(stereo, left_img, right_img)
        #     slider.create_slider()
        #
        # disparity_img = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
        # norm_disparity_img = cv2.normalize(disparity_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
        #                                    dtype=cv2.CV_32F)

        left_img_blurred = cv2.GaussianBlur(left_img, (23, 23), 0)
        difference = cv2.absdiff(left_img_blurred, reference_img)
        # diff_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)
        thresh_gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        # cutout = cv2.cvtColor(left_img * mask[:, :, None], cv2.COLOR_BGR2RGB)
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.asarray(imutils.grab_contours(contours))
        cv2.drawContours(left_img, contours, -1, (0, 255, 0), 3)

        if len(contours) > 0:
            cx, cy = calculate_centroid(contours)
            cv2.circle(left_img, (cx, cy), 20, (255, 0, 0), -1)


        cv2.imshow('Images', left_img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # export_pointcloud(disparity_map=disparity_img, colors=left_img, Q=Q_scaled)


if __name__ == '__main__':
    main()
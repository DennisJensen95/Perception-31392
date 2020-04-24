import glob
import numpy as np
import imutils
# from skimage.metrics import structural_similarity as ssim
from lib.calibration import Calibration
from lib.construct3D import *
from lib.tracking import *
from lib.kalman3d import *
import cv2
from lib.sliderProgram import Slider
import matplotlib.pyplot as plt
import imutils


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
        Cal.save_class('ClassDataSaved')
    else:
        Cal.load_class('ClassDataSaved')

    Kal = KalmanFilter()
    last_centroid = [0, 0]

    baseline = 0.12
    focal_length = Cal.Q[2, 3]
    down_sample_ratio = 0.4

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


    # object for background subtraction
    fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=500, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    images_conveyor = images_conveyor[0:-1]  # start from first box with 80:

    # left_img, _ = Cal.remapImagesStereo(images_conveyor[0], random=False, debug=False)
    # reference_img = cv2.GaussianBlur(left_img, (11, 11), 0)  # reference image to be used for tracking

    for images in images_conveyor[0:]:
        left_img, right_img = Cal.remapImagesStereo(images, random=False, debug=False)
        # images = np.concatenate((left_img, right_img), axis=1)

        left_img = downsample_image(left_img, down_sample_ratio)
        right_img = downsample_image(right_img, down_sample_ratio)

        # plt.imshow(left_img[80:-1, 100:-1])
        # plt.show()

        if debug_disp_slider:
            slider = Slider(stereo, left_img, right_img)
            slider.create_slider()

        # Compute disparity image
        disparity_img = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

        # Apply background subtraction
        mask = fgbg.apply(left_img)

        # Open and close mask to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)

        # Calculate contours
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.asarray(imutils.grab_contours(contours))
        # cv2.drawContours(left_img, contours, -1, (0, 255, 0), 3)

        # Do calculation if an object was found
        if len(contours) > 0:
            centroid = calculate_centroid(contours)
            assert(disparity_img.shape[:2] == left_img.shape[:2]), f'Disparity img shape: {disparity_img.shape[:2]}, ' \
                                                           f'Left img shape: {left_img.shape[:2]}'

            if centroid:
                """Kalman predict and update"""
                # extract centroid x,y
                cx, cy = centroid

                # plot measurement on image
                left_img = Kal.plot_pos(contours[-1], centroid, left_img, pred=False)

                # find Z-coordinate from disparity image
                cz = construct_z_coordinate(disparity_img[cy, cx], baseline, focal_length * down_sample_ratio)

                # append to centroid list
                centroid_pos = [centroid[0], centroid[1], cz]

                # if a drastic change is detected, it must be a new object detected
                if np.abs(last_centroid[0]-centroid[0]) > 125:
                    print(f'Reset kalman filter')
                    Kal.reset_kalman()

                centroid_pred = Kal.kalman(centroid_pos, update=True)

                # plot kalman prediction on image
                left_img = Kal.plot_pos(contours[-1], centroid_pred, left_img, pred=True)

                # update last centroid
                last_centroid = centroid

                # Display text on screen
                measurement_string = f'Measurement: (x, y, z) = ({centroid[0]},{centroid[1]},{cz:.2f})'
                cv2.putText(left_img, measurement_string, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
                prediction_string = f'Prediction: (x, y, z) = ({centroid_pred[0][0]:.2f},{centroid_pred[1][0]:.2f},{centroid_pred[2][0]:.2f})'
                cv2.putText(left_img, prediction_string, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
            else:
                """Kalman Predict"""
                centroid_pred = Kal.kalman(update=False)
                left_img = Kal.plot_pos(contours[-1], centroid_pred, left_img, pred=True)

                # Display on screen
                prediction_string = f'Prediction: (x, y, z) = ({centroid_pred[0][0]:.2f},{centroid_pred[1][0]:.2f},{centroid_pred[2][0]:.2f})'
                cv2.putText(left_img, prediction_string, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))


        cv2.imshow('Images', left_img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # export_pointcloud(disparity_map=disparity_img, colors=left_img, Q=Q_scaled)


if __name__ == '__main__':
    main()
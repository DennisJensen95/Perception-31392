import glob
import numpy as np
import imutils
# from skimage.metrics import structural_similarity as ssim
from lib.calibration import Calibration
from lib.construct3D import *
from lib.tracking import *
import cv2
from lib.sliderProgram import Slider
import matplotlib.pyplot as plt
import imutils
from lib.fastRCNNPretrained import object_detection_api


def main():
    images_left = sorted(glob.glob("data/Stereo_calibration_images/left*.png"))
    images_right = sorted(glob.glob("data/Stereo_calibration_images/right*.png"))
    images_left_conveyor = sorted(glob.glob("data/stereo_conveyor_with_occlusions/left/*.png"))
    images_right_conveyor = sorted(glob.glob("data/stereo_conveyor_with_occlusions/right/*.png"))
    images_conveyor = np.asarray([images_left_conveyor, images_right_conveyor]).T
    images = np.asarray([images_left, images_right]).T

    calibrate = False
    debug_disp_slider = False
    preTrainedFastRcnn = False

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

    baseline = 0.12
    focal_length = Cal.Q[2, 3]


    # Stereo Class
    min_disparity = -100  # 1
    num_disparity = 1  # 160
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


    objects_to_detect = ['cup', 'book']
    down_sample_ratio = 0.4

    fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=500, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    images_conveyor = images_conveyor[0:-1]  # start from first box with 80:
    last_centroid = [0, 0]

    measurement = np.zeros(Kal.meausure_num, np.float32)
    for images in images_conveyor[0:]:
        left_img, right_img = Cal.remapImagesStereo(images, random=False, debug=False)
        # images = np.concatenate((left_img, right_img), axis=1)

        left_img = downsample_image(left_img, down_sample_ratio)
        right_img = downsample_image(right_img, down_sample_ratio)

        # plt.imshow(left_img[80:-1, 100:-1])
        # plt.show()

        if debug_disp_slider:
            slider = Slider(stereo, right_img, left_img)
            slider.create_slider()

        disparity_img = stereo.compute(right_img, left_img).astype(np.float32) / 16.0
        # norm_disparity_img = cv2.normalize(disparity_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
        #                                    dtype=cv2.CV_32F

        mask = fgbg.apply(right_img)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)

        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.asarray(imutils.grab_contours(contours))
        # cv2.drawContours(left_img, contours, -1, (0, 255, 0), 3)

        if len(contours) > 0:
            centroid = calculate_centroid(contours)
            assert(disparity_img.shape[:2] == right_img.shape[:2]), f'Disparity img shape: {disparity_img.shape[:2]}, ' \
                                                           f'Left img shape: {right_img.shape[:2]}'

            if centroid:
                """Kalman predict and update"""
                cx, cy = centroid
                right_img = Kal.plot_pos(contours[-1], centroid, right_img, pred=False)

                z_coor = disparity_img[cy, cx]

                pos_string = f'(x, y, z) = ({centroid[0]},{centroid[1]},{z_coor})'
                centroid_pos = np.array([centroid[0], centroid[1], z_coor], np.float32)

                if np.abs(last_centroid[0]-centroid[0]) > 125:
                    print(f'Reset kalman filter')
                    Kal.reset_kalman()

                centroid_pred = Kal.kalman_cv2(centroid_pos, update=True)

                right_img = Kal.plot_pos(contours[-1], centroid_pred, right_img, pred=True)

                last_centroid = centroid

                # Display text on screen
                measurement_string = f'Measurement: (x, y, z) = ({centroid[0]},{centroid[1]},{z_coor:.2f})'
                cv2.putText(right_img, measurement_string, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
                prediction_string = f'Prediction: (x, y, z) = ({centroid_pred[0][0]:.2f},{centroid_pred[1][0]:.2f},{centroid_pred[2][0]:.2f})'
                cv2.putText(right_img, prediction_string, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))

                last_found_contour = contours[-1]
        else:
            print("predicting")
            """Kalman Predict"""
            centroid_pred = Kal.kalman_cv2(update=False)
            right_img = Kal.plot_pos(last_found_contour, centroid_pred, right_img, pred=True)

            # Display text on screen
            prediction_string = f'Prediction: (x, y, z) = ({centroid_pred[0][0]:.2f},{centroid_pred[1][0]:.2f},{centroid_pred[2][0]:.2f})'
            cv2.putText(right_img   , prediction_string, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))

        if preTrainedFastRcnn:
            right_img = right_img[80:-1, 100:-1]
            img_fastRCNN = object_detection_api(right_img, objects_to_detect, threshold=0.8)

        cv2.imshow('Images', right_img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # export_pointcloud(disparity_map=disparity_img, colors=right_img, Q=Q_scaled)


if __name__ == '__main__':
    main()
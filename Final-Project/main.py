import glob
from lib.calibration import Calibration
from lib.construct3D import *
from lib.tracking import *
import cv2
from lib.sliderProgram import Slider
import torch
import imutils
from lib.fastRCNNPretrained import object_detection_api
from lib.Classification import Classifier, YOLOClassifier, NeuralNetClassifier


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    images_left = sorted(glob.glob("data/Stereo_calibration_images/left*.png"))
    images_right = sorted(glob.glob("data/Stereo_calibration_images/right*.png"))
    images_left_conveyor = sorted(glob.glob("data/Stereo_conveyor_with_occlusions/left/*.png"))
    images_right_conveyor = sorted(glob.glob("data/Stereo_conveyor_with_occlusions/right/*.png"))
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

    track = Tracking(running_mean_num=1)
    clf = NeuralNetClassifier()
    clf.load_model(device, eval=True)

    # path_pca = './Results/Saved_SVM_Models/PCA_transform.sav'
    # path_clf = './Results/Saved_SVM_Models/PCA_final_open_image.sav'
    # classification = Classifier(path_clf, path_pca, device)
    # classification = YOLOClassifier()

    baseline = 0.12 # m
    down_sample_ratio = 0.4
    focal_length = Cal.Q[2, 3] * down_sample_ratio

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


    objects_to_detect = ['cup', 'book']


    images_conveyor = images_conveyor[0:-1]  # start from first box with 80:
    last_centroid = [0, 0]
    last_centroid_pred = [[0], [0], [0], [0], [0], [0]]
    area_threshold = 0.55
    biggest_area = 0

    # cutout of the board covering the conveyor belt
    cutout = cv2.imread("occlusion_ref_cutout.png")
    cutout = cv2.cvtColor(cutout, cv2.COLOR_BGR2GRAY)
    # create binary mask of the cutout
    _, thresh = cv2.threshold(cutout, 240, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(thresh)
    plt.show()

    predict_only = False
    img_array = []
    left_files = ["data/Stereo_conveyor_without_occlusions/left/*.png", "data/Stereo_conveyor_with_occlusions/left/*.png"]
    right_files = ["data/Stereo_conveyor_without_occlusions/right/*.png", "data/Stereo_conveyor_with_occlusions/right/*.png"]
    thickness = 2
    scale_text = 0.55
    last_z_coor = 0
    for i in range(len(left_files)):


        fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=500, detectShadows=False)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        images_left_conveyor = sorted(glob.glob(left_files[i]))
        images_right_conveyor = sorted(glob.glob(right_files[i]))
        images_conveyor = np.asarray([images_left_conveyor, images_right_conveyor]).T

        for images in images_conveyor[0:]:
            left_img, right_img = Cal.remapImagesStereo(images, random=False, debug=False)
            # images = np.concatenate((left_img, right_img), axis=1)

            left_img = downsample_image(left_img, down_sample_ratio)
            right_img = downsample_image(right_img, down_sample_ratio)

            if debug_disp_slider:
                slider = Slider(stereo, left_img, right_img)
                slider.create_slider()

            disparity_img = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
            # norm_disparity_img = cv2.normalize(disparity_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
            #                                    dtype=cv2.CV_32F

            mask = fgbg.apply(left_img)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)

            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = np.asarray(imutils.grab_contours(contours))
            # cv2.drawContours(left_img, contours, -1, (0, 255, 0), 3)

            if len(contours) > 0:
                centroid = track.calculate_centroid(contours)
                assert(disparity_img.shape[:2] == left_img.shape[:2]), f'Disparity img shape: {disparity_img.shape[:2]}, ' \
                                                               f'Left img shape: {left_img.shape[:2]}'

                if i == 0:
                    predict_only = track.check_area_to_small(contours[-1], area_threshold)
                else:
                    predict_only = track.check_object_occlusion(mask, thresh)

                print(predict_only)

                if centroid and not predict_only:
                    """Kalman predict and update"""
                    cx, cy = centroid

                    # Save images
                    crop_img = track.crop_image_rectangle(left_img, contours[-1], images[0], save=True)
                    label_predict = clf.classify_img(crop_img, device)

                    if preTrainedFastRcnn:
                        resp = object_detection_api(np.array(crop_img), objects_to_detect, threshold=0.5, label=True)
                        if isinstance(resp, tuple):
                            label, confi = resp
                        else:
                            label = None
                            confi = None
                        print(f'Detected: {label}: Confidence: {confi}')

                    left_img = track.plot_pos(contours[-1], centroid, left_img, pred=False)

                    z_coor = disparity_img[cy, cx]
                    last_z_coor = z_coor

                    centroid_pos = np.array([centroid[0], centroid[1], z_coor], np.float32)

                    if np.abs(last_centroid[0]-centroid[0]) > 125:
                        print(f'Reset kalman filter')
                        track.reset_kalman()

                    centroid_pred = track.kalman_cv2(centroid_pos, update=True)

                    left_img = track.plot_pos(contours[-1], centroid_pred, left_img, pred=True)

                    (x, y, w, h) = cv2.boundingRect(contours[-1])
                    cv2.putText(left_img, label_predict, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, scale_text, (0, 0, 255), thickness=thickness)

                    # Display text on screen
                    z_coor_m = construct_z_coordinate(z_coor, baseline, focal_length, last_z_coor)
                    z_coor_pred_m = construct_z_coordinate(centroid_pred[2][0], baseline, focal_length, last_centroid_pred[2][0])
                    z_coor_pred_m_s = construct_z_coordinate(centroid_pred[5][0], baseline, focal_length, last_centroid_pred[5][0])
                    measurement_string = f'Measurement: (x, y, z) = ({centroid[0]},{centroid[1]},{z_coor_m:.2f})'
                    cv2.putText(left_img, measurement_string, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, scale_text, (0, 0, 255), thickness=thickness)
                    prediction_string = f'Prediction: (x, y, z) = ({centroid_pred[0][0]:.2f},{centroid_pred[1][0]:.2f},{z_coor_pred_m:.2f})'
                    cv2.putText(left_img, prediction_string, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, scale_text, (255, 0, 0), thickness=thickness)
                    prediction_string = f'Prediction: (x_vel, y_vel, z_vel) = ({centroid_pred[3][0]:.2f},{centroid_pred[4][0]:.2f},{z_coor_pred_m_s:.2f})'
                    cv2.putText(left_img, prediction_string, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, scale_text,
                                (255, 0, 0),
                                thickness=thickness)
                    last_centroid = centroid
                    last_centroid_pred = centroid_pred

                    last_found_contour = contours[-1]
                    predict_only = False
                else:
                    if np.abs(last_centroid[0]-centroid[0]) > 125:
                        print(f'Reset kalman filter')
                        track.reset_kalman()
            else:
                predict_only = True


            if predict_only:
                """Kalman Predict"""
                centroid_pred = track.kalman_cv2(update=False)
                left_img = track.plot_pos(last_found_contour, centroid_pred, left_img, pred=True)
                (x, y, w, h) = cv2.boundingRect(last_found_contour[-1])
                cv2.putText(left_img, label_predict, (int(centroid_pred[0][0])-int(w/2),
                                                      int(centroid_pred[1][0])-int(h/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, scale_text, (0, 0, 255), thickness=thickness)


                # Display text on screen
                z_coor_pred_m = construct_z_coordinate(centroid_pred[2][0], baseline, focal_length,
                                                       last_centroid_pred[2][0])
                z_coor_pred_m_s = construct_z_coordinate(centroid_pred[5][0], baseline, focal_length,
                                                         last_centroid_pred[5][0])
                prediction_string = f'Prediction: (x, y, z) = ({centroid_pred[0][0]:.2f},{centroid_pred[1][0]:.2f},{z_coor_pred_m:.2f})'
                cv2.putText(left_img, prediction_string, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, scale_text, (255, 0, 0),
                            thickness=thickness)
                prediction_string = f'Prediction: (x_vel, y_vel, z_vel) = ({centroid_pred[3][0]:.2f},{centroid_pred[4][0]:.2f},{z_coor_pred_m_s:.2f})'
                cv2.putText(left_img, prediction_string, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, scale_text,
                            (255, 0, 0),
                            thickness=thickness)

                last_centroid_pred = centroid_pred

            img_array.append(left_img)
            cv2.imshow('Images', left_img)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

    height, width = (left_img.shape[:2])
    size = (width, height)

    out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()

if __name__ == '__main__':
    main()
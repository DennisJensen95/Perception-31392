import numpy as np
import cv2
import glob
import os
import re
import matplotlib.pyplot as plt


class Calibration:

    def __init__(self, images, nb_vertical, nb_horizontal):
        """
        :param images: A list with stereo images left and right
        :param nb_vertical: Number of vertical corners
        :param nb_horizontal: Number of horizontal corners
        """
        print(f'Opencv version: {cv2.__version__}')
        self.images = images
        assert(np.asarray(images).shape[1] == 2), "Please make sure you have a left and right image in the list"

        self.nb_vertical = nb_vertical
        self.nb_horizontal = nb_horizontal

        self.img_shape = None

        # Calibration data
        self.left_camera_calibration_data = None
        self.right_camera_calibration_data = None

        self.imgpoints_l = None
        self.imgpoints_r = None
        self.objpoints = None

        # Stereo
        self.leftMapX, self.leftMapY = None, None
        self.rightMapX, self.rightMapY = None, None

        self.optimal_camMtx1_stereo, self.optimal_camMtx2_stereo = None, None

        self.rectification_data = None

        self.stereo_calibration_data = None

        # Status variables
        self.calibrated_cameras = False

        # Calibration criterias
        self.stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        # Flags
        self.stereo_flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_FOCAL_LENGTH |
                 cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 |
                 cv2.CALIB_FIX_K6)

    def calibrateCamera(self, debug=False, fisheye=False):
        """
        :return: Saves calibration parameters inside class variable
        """
        # Sub pixels criteria
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

        # Prepare data objects
        if fisheye:
            objp = np.zeros((1, self.nb_horizontal * self.nb_vertical, 3), np.float32)
            objp[0,:,:2] = np.mgrid[0:self.nb_vertical, 0:self.nb_horizontal].T.reshape(-1, 2)
        else:
            objp = np.zeros((self.nb_horizontal * self.nb_vertical, 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.nb_vertical, 0:self.nb_horizontal].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints_left = []  # 2d points in image plane.
        imgpoints_right = []  # 2d points in image plane.
        img_shape = None
        for images in self.images:
            # Read images
            left_img = cv2.imread(images[0])
            right_img = cv2.imread(images[1])

            # Greyscale
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            ret_l, corners_l = cv2.findChessboardCorners(gray_left, (self.nb_vertical, self.nb_horizontal), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_right, (self.nb_vertical, self.nb_horizontal), None)

            # Placeholder for the number of images
            objpoints.append(objp)
            if ret_l:
                corners_l = cv2.cornerSubPix(gray_left, corners_l, (11, 11), (-1, -1), criteria)

                imgpoints_left.append(corners_l)
                # Draw and display the corners
                if debug:
                    img = cv2.drawChessboardCorners(left_img, (self.nb_vertical, self.nb_horizontal), corners_l, ret_l)
                    cv2.imshow('img', img)
                    cv2.waitKey(10)

            if ret_r:
                corners_r = cv2.cornerSubPix(gray_right, corners_r, (11, 11), (-1, -1), criteria)

                imgpoints_right.append(corners_r)

                if debug:
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(right_img, (self.nb_vertical, self.nb_horizontal), corners_r, ret_r)
                    cv2.imshow('img', img)
                    cv2.waitKey(10)

            img_shape = gray_left.shape[::-1]

        self.img_shape = img_shape

        # Close windows if
        if debug:
            cv2.destroyAllWindows()

        assert(img_shape != None), "img_shape was not determined"
        # Calibration
        # tuple contains (retval, cameraMatrix, distCoeffs, rvecs, tvecs)
        print("Calculating camera matrix and distortion coefficients for left and right.")
        if fisheye:
            self.left_camera_calibration_data = cv2.fisheye.calibrate(
                objpoints,
                imgpoints_left,
                img_shape, None, None)

            self.right_camera_calibration_data = cv2.fisheye.calibrate(
                objpoints,
                imgpoints_right,
                img_shape, None, None)
        else:
            self.left_camera_calibration_data = cv2.calibrateCamera(objpoints, imgpoints_left, img_shape, None, None)
            self.right_camera_calibration_data = cv2.calibrateCamera(objpoints, imgpoints_right, img_shape, None, None)

        if debug:
            print("Testing if distortion correction makes sense")
            # Test calibration
            print(self.images[0][0])
            img = cv2.imread(self.images[0][0])
            h, w = img.shape[:2]
            print(self.left_camera_calibration_data[1])
            print(self.left_camera_calibration_data[2])
            print((h,w))


            newcameramtx_left, roi_left = cv2.getOptimalNewCameraMatrix(self.left_camera_calibration_data[1],
                                                                        self.left_camera_calibration_data[2],
                                                                        (w, h), 1, (w, h))
            print(roi_left)
            dst = cv2.undistort(img,
                                self.left_camera_calibration_data[1],
                                self.left_camera_calibration_data[2],
                                None,
                                newcameramtx_left)

            print(dst.shape)
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 18))
            ax[0].imshow(img[..., [2, 1, 0]])
            ax[0].set_title('Original image')
            # crop the image
            x, y, w, h = roi_left
            # dst = dst[y:y + h, x:x + w]
            # print(dst.shape)
            ax[1].imshow(dst)
            ax[1].set_title('Undistorted image')
            plt.show()

        self.calibrated_cameras = True
        self.objpoints = objpoints
        self.imgpoints_l = imgpoints_left
        self.imgpoints_r = imgpoints_right

    def stereoCalibration(self, debug=False):
        """
        :return: Saves calibration parameters for stereo calibration to the class
        """
        assert self.calibrated_cameras, "Please do camera calibration before stereo calibration"

        # Two new camera matrices and distortion  coefficients from the stereo camera calibration and also Rotation
        # and translation
        self.stereo_calibration_data = cv2.stereoCalibrate(self.objpoints,
                                                           self.imgpoints_l,
                                                           self.imgpoints_r,
                                                           self.left_camera_calibration_data[1],
                                                           self.left_camera_calibration_data[2],
                                                           self.right_camera_calibration_data[1],
                                                           self.right_camera_calibration_data[2],
                                                           self.img_shape,
                                                           criteria =self.stereocalib_criteria,
                                                           flags=self.stereo_flags)

        # New optimal camera matrix

        self.optimal_camMtx1_stereo, self.roi1_stereo  = cv2.getOptimalNewCameraMatrix(self.stereo_calibration_data[1],
                                                                          self.stereo_calibration_data[2],
                                                                          self.img_shape,
                                                                          0,
                                                                          self.img_shape)

        self.optimal_camMtx2_stereo, self.roi2_stereo = cv2.getOptimalNewCameraMatrix(self.stereo_calibration_data[3],
                                                                          self.stereo_calibration_data[4],
                                                                          self.img_shape,
                                                                          0,
                                                                          self.img_shape)

        self.rectification_data = cv2.stereoRectify(self.stereo_calibration_data[1],    # Camera matrix 1
                                                    self.stereo_calibration_data[2],    # Distortion coef camera 1
                                                    self.stereo_calibration_data[3],    # Camera matrix 2
                                                    self.stereo_calibration_data[4],    # Distortion coef camera 2
                                                    self.img_shape,
                                                    self.stereo_calibration_data[5],    # Rotation matrix
                                                    self.stereo_calibration_data[6],    # Translation vec
                                                    None, None, None, None, None,
                                                    alpha=0)

        self.leftMapX, self.leftMapY = cv2.initUndistortRectifyMap(self.stereo_calibration_data[1],
                                                                     self.stereo_calibration_data[2],
                                                                     self.rectification_data[0],
                                                                     self.rectification_data[2],
                                                                     self.img_shape, cv2.CV_32FC1)

        self.rightMapX, self.rightMapY = cv2.initUndistortRectifyMap(self.stereo_calibration_data[3],
                                                                     self.stereo_calibration_data[4],
                                                                     self.rectification_data[1],
                                                                     self.rectification_data[3],
                                                                     self.img_shape, cv2.CV_32FC1)

    def remapImagesStereo(self, images=None, random=False, debug=False):
        """
        :param images: Left and right image in that order in a list
        :param random: If true choose random left and right image from calibration data to remap
        """

        if random:
            idx = np.random.randint(0, len(self.images))
            image_l = self.images[idx][0]
            image_r = self.images[idx][1]
        else:
            assert(len(images) == 2), "Please input images or use random from calibration data images"
            image_l = images[0]
            image_r = images[1]

        img_l = cv2.imread(image_l)
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

        img_r = cv2.imread(image_r)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if debug:
            img_l_undis = cv2.undistort(img_l,
                                        self.stereo_calibration_data[1],
                                        self.stereo_calibration_data[2],
                                        None,
                                        self.optimal_camMtx1_stereo)

            img_r_undis = cv2.undistort(img_r,
                                        self.stereo_calibration_data[3],
                                        self.stereo_calibration_data[4],
                                        None,
                                        self.optimal_camMtx2_stereo)

            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))
            ax[0, 0].imshow(img_l)
            ax[0, 0].set_title('Original image left')
            # crop the image
            x, y, w, h = self.roi1_stereo
            dst = img_l_undis[y:y + h, x:x + w]
            ax[1, 0].imshow(dst)
            ax[1, 0].set_title('Undistorted image left')

            ax[0, 1].imshow(img_r)
            ax[0, 1].set_title('Original image right')

            x, y, w, h = self.roi2_stereo
            dst = img_r_undis[y:y + h, x:x + w]

            ax[1, 1].imshow(dst)
            ax[1, 1].set_title('Undistorted image right')
            plt.show()

        # remap
        print(self.rightMapX.shape)
        print(self.leftMapX.shape)
        imglCalRect = cv2.remap(img_l, self.leftMapX, self.leftMapY, cv2.INTER_LINEAR)
        imgrCalRect = cv2.remap(img_r, self.rightMapX, self.rightMapY, cv2.INTER_LINEAR)

        if debug:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 18))
            ax[0].imshow(img_l)
            ax[0].set_title('Rectified image left')
            # crop the image
            ax[1].imshow(dst)
            ax[1].set_title('Rectified image right')
            plt.show()

        if debug:
            numpyHorizontalCalibRect = np.hstack((imglCalRect, imgrCalRect))
            ### SHOW RESULTS ###
            # calculate point arrays for epipolar lines
            lineThickness = 5
            lineColor = [0, 255, 0]
            numLines = 20
            interv = round(self.img_shape[0] / numLines)
            x1 = np.zeros((numLines, 1))
            y1 = np.zeros((numLines, 1))
            x2 = np.full((numLines, 1), (4 * self.img_shape[1]))
            for jj in range(0, numLines):
                y1[jj] = jj * interv
            y2 = y1

            for jj in range(0, numLines):
                cv2.line(numpyHorizontalCalibRect, (x1[jj], y1[jj]), (x2[jj], y2[jj]),
                         lineColor, lineThickness)

            cv2.namedWindow("calibRect", cv2.WINDOW_NORMAL)
            cv2.imshow("calibRect", numpyHorizontalCalibRect)
            k = cv2.waitKey(0)

            if k == 27:
                cv2.destroyAllWindows()

        return imglCalRect, imgrCalRect

    def save_remapping_instance(self, dir_name):
        """
        Load remapping
        :param file_name:
        :return:
        """
        np.savetxt(dir_name + '/leftMapX', self.leftMapX, delimiter=',')
        np.savetxt(dir_name + '/leftMapY', self.leftMapY, delimiter=',')
        np.savetxt(dir_name + '/rightMapX', self.rightMapX, delimiter=',')
        np.savetxt(dir_name + '/rightMapY', self.rightMapY, delimiter=',')

    def load_remapping_instance(self, dir_name):
        """
        Load remapping maps
        :param dir_name:
        :return:
        """
        self.leftMapX = np.array(np.loadtxt(dir_name + '/leftMapX', delimiter=','), dtype='float32')
        self.leftMapY = np.loadtxt(dir_name + '/leftMapY', delimiter=',', dtype='float32')
        self.rightMapX = np.loadtxt(dir_name + '/rightMapX', delimiter=',', dtype='float32')
        self.rightMapY = np.loadtxt(dir_name + '/rightMapY', delimiter=',', dtype='float32')


if __name__ == '__main__':
    test_images = [["Left", "Right"], ["Left_2", "Right_2"], ["Left_3", "Right_3"]]
    Cal = Calibration(test_images, 6, 9)

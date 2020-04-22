import cv2
import numpy as np


class Slider():

    def __init__(self, stereo, image_l, image_r):
        self.image_l = image_l
        self.image_r = image_r
        self.stereo = stereo

    def update(self, sliderValue = 0):

        self.stereo.setBlockSize(
                cv2.getTrackbarPos('blockSize', 'Disparity'))
        self.stereo.setUniquenessRatio(
                cv2.getTrackbarPos('uniquenessRatio', 'Disparity'))
        self.stereo.setSpeckleRange(
                cv2.getTrackbarPos('speckleRange', 'Disparity'))
        self.stereo.setDisp12MaxDiff(
                cv2.getTrackbarPos('disp12MaxDiff', 'Disparity'))
        self.stereo.setMinDisparity(
            cv2.getTrackbarPos('minDisparity', 'Disparity'))
        self.stereo.setSpeckleWindowSize(
            cv2.getTrackbarPos('speckleWindowSize', 'Disparity'))
        self.stereo.setNumDisparities(
            cv2.getTrackbarPos('numDisparities', 'Disparity'))
        self.stereo.setPreFilterCap(
            cv2.getTrackbarPos('preFilterCap', 'Disparity'))
        # P1 = 3 * 8 * cv2.getTrackbarPos('blockSize', 'Disparity')
        # P2 = 3 * 32 * cv2.getTrackbarPos('blockSize', 'Disparity')
        # self.stereo.setP1(P1)
        # self.stereo.setP2(P2)
        self.stereo.setP1(
            cv2.getTrackbarPos('P1', 'Disparity'))
        self.stereo.setP2(
            cv2.getTrackbarPos('P2', 'Disparity'))

        disparity = self.stereo.compute(self.image_l, self.image_r).astype(np.float32) / 16.0
        cv2.imshow('Disparity',
                   (disparity - self.stereo.getMinDisparity()) / cv2.getTrackbarPos('numDisparities', 'Disparity'))

    def create_slider(self):
        cv2.namedWindow('Disparity')
        cv2.createTrackbar('blockSize', 'Disparity', self.stereo.getBlockSize(), 50, self.update)
        cv2.createTrackbar('uniquenessRatio', 'Disparity', self.stereo.getUniquenessRatio(), 50, self.update)
        cv2.createTrackbar('speckleWindowSize', 'Disparity', self.stereo.getSpeckleWindowSize(), 200, self.update)
        cv2.createTrackbar('speckleRange', 'Disparity', self.stereo.getSpeckleRange(), 50, self.update)
        cv2.createTrackbar('disp12MaxDiff', 'Disparity', self.stereo.getDisp12MaxDiff(), 250, self.update)
        cv2.createTrackbar('minDisparity', 'Disparity', self.stereo.getMinDisparity(), 250, self.update)
        cv2.createTrackbar('numDisparities', 'Disparity', self.stereo.getNumDisparities(), 250, self.update)
        cv2.createTrackbar('preFilterCap', 'Disparity', self.stereo.getPreFilterCap(), 250, self.update)
        cv2.createTrackbar('P1', 'Disparity', self.stereo.getP1(), 1000, self.update)
        cv2.createTrackbar('P2', 'Disparity', self.stereo.getP2(), 5000, self.update)

        self.update()
        cv2.waitKey()
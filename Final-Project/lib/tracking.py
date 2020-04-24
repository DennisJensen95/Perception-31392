import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from numpy.linalg import inv, pinv


class Tracking:
    def __init__(self, running_mean_num=3):
        # The initial state (9x1).
        self.state_num = 9
        self.meausure_num = 3
        self.kal = cv2.KalmanFilter(self.state_num, self.meausure_num)

        self.biggest_area = 0
        self.area = 0

        self.centroids_x_y = []
        self.running_mean_num = running_mean_num

        self.reset_kalman()

    def reset_kalman(self):
        self.biggest_area = 0

        self.x = np.array([[0],
                           [0],
                           [0],
                           [0],
                           [0],
                           [0],
                           [0],
                           [0],
                           [0]], np.float32)

        # The initial uncertainty (9x9).
        self.P = np.eye(9) * 100

        # The external motion (9x1).
        self.u = np.array([[0],
                           [0],
                           [0],
                           [0],
                           [0],
                           [0],
                           [0],
                           [0],
                           [0]], np.float32)

        # The transition matrix (9x9).
        delta_t = 0.1
        self.F = np.array([[1, 0, 0, delta_t, 0, 0, 1 / 2 * delta_t ** 2, 0, 0],
                           [0, 1, 0, 0, delta_t, 0, 0, 1 / 2 * delta_t ** 2, 0],
                           [0, 0, 1, 0, 0, delta_t, 0, 0, 1 / 2 * delta_t ** 2],
                           [0, 0, 0, 1, 0, 0, delta_t, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, delta_t, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, delta_t],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1]], np.float32)

        # The observation matrix (3x9).
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0]], np.float32)

        # The measurement uncertainty
        self.R = 1

        # Idendity matrix
        self.I = np.eye(9)

        self.x_pred = None
        self.P_pred = None
        self.x_update = None
        self.P_update = None


        self.kal.processNoiseCov = np.identity(
            self.state_num, np.float32) * 1e-5
        self.kal.measurementNoiseCov = np.identity(
            self.meausure_num, np.float32) * 1e-2
        self.kal.errorCovPost = np.identity(
            self.state_num, np.float32)
        self.kal.transitionMatrix = self.F
        self.kal.measurementMatrix = self.H

    def update(self, x, P, Z):
        """
        :param x: State vector
        :param P: Uncertainty Matrix
        :param Z: Measurement observation
        :param H: Observation Matrix
        :param R: Measurement uncertainty
        :return: updated state vector, updated uncertainty matrix
        """

        I = np.eye(P.shape[0])
        y = Z - self.H @ x
        S = self.H @ P @ np.transpose(self.H) + self.R
        K = P @ np.transpose(self.H) @ np.linalg.pinv(S)
        x_update = x + K @ y
        P_update = (I - K @ self.H) @ P

        return x_update, P_update

    def predict(self, x, P):
        """
        :param x: State vector
        :param P: Uncertainty Matrix
        :param F: Transition Matrix
        :param u: External motion
        :return: predicted state vector, predicted uncertainty matrix
        """
        x_pred = self.F @ x + self.u
        P_pred = self.F @ P @ np.transpose(self.F)

        return x_pred, P_pred

    def kalman(self, centroid=None, update=True):
        if update:
            centroid = np.array([[centroid[0, 0]], [centroid[0, 1]], [centroid[0, 2]]])
            self.x_pred, self.P_pred = self.predict(self.x, self.P)
            self.x_update, self.P_update = self.update(self.x_pred, self.P_pred, centroid)
            self.x, self.P = self.x_update, self.P_update
            # Prediction
            center_pred = (self.x_pred[0], self.x_pred[1], self.x_pred[2])
        else:
            # Predict
            self.x_pred, self.P_pred = self.predict(self.x, self.P)
            # Update x and P to prediction
            self.x, self.P = self.x_pred, self.P_pred
            # Save prediction
            center_pred = (self.x_pred[0], self.x_pred[1], self.x_pred[2])

        return center_pred

    def kalman_cv2(self, centroid=None, update=True):
        if update:
            self.x_pred = self.kal.predict()
            self.kal.correct(centroid)
            # Prediction
            center_pred = (self.x_pred[0], self.x_pred[1], self.x_pred[2])
        else:
            # Predict
            self.x_pred = self.kal.predict()
            # Save prediction
            center_pred = (self.x_pred[0], self.x_pred[1], self.x_pred[2])

        return center_pred

    def plot_pos(self, contour, centroid, img, pred=False):
        if not pred:
            cx, cy = centroid
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.circle(img, (cx, cy), radius=4, color=(0, 0, 255), thickness=5)
        else:
            cx, cy, z = centroid
            cx, cy = int(cx), int(cy)
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img, (cx - int(w/2), cy - int(h/2)), (cx + int(w/2), cy + int(h/2)), (255, 0, 0), 3)
            cv2.circle(img, (cx, cy), radius=4, color=(255, 0, 0), thickness=5)

        return img

    def get_area_of_box(self, contour):
        (x, y, w, h) = cv2.boundingRect(contour)
        return w*h

    def check_area_to_small(self, contour, threshold):
        self.area = self.get_area_of_box(contour)

        if self.area > self.biggest_area:
            self.biggest_area = self.area
            return False
        elif self.area < self.biggest_area * threshold:
            return True


    def running_mean_centroid(self):
        centroid_pos_x_y = np.mean(np.asarray(self.centroids_x_y), axis=0).astype(int)
        return centroid_pos_x_y

    def calculate_centroid(self, contours):
        cnt = contours[-1]
        M = cv2.moments(cnt)
        try:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        except ZeroDivisionError as e:
            print(e)
            return False
        centroid = [cx, cy]
        self.centroids_x_y.insert(0, centroid)
        self.centroids_x_y = self.centroids_x_y[:self.running_mean_num]
        return cx, cy
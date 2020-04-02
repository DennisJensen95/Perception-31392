import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils


def update(x, P, Z, H, R):
    """
    :param x: State vector
    :param P: Uncertainty Matrix
    :param Z: Measurement observation
    :param H: Observation Matrix
    :param R: Measurement uncertainty
    :return: updated state vector, updated uncertainty matrix
    """

    I = np.eye(P.shape[0])
    y = Z - H @ x
    S = H @ P @ np.transpose(H) + R
    K = P @ np.transpose(H) @ np.linalg.pinv(S)
    x_k = x + K @ y
    P_k = (I - K @ H) @ P

    return x_k, P_k


def predict(x, P, F, u):
    """
    :param x: State vector
    :param P: Uncertainty Matrix
    :param F: Transition Matrix
    :param u: External motion
    :return: predicted state vector, predicted uncertainty matrix
    """
    x_k = F @ x + u
    P_k = F @ P @ np.transpose(F)

    return x_k, P_k
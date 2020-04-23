import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from numpy.linalg import inv, pinv


"""
FEATURE TRACKING
"""


def calculate_centroid(contours):
    cnt = contours[-1]
    M = cv2.moments(cnt)

    cx = int(M['m01'] / M['m00'])
    cy = int(M['m10'] / M['m00'])

    return cx, cy

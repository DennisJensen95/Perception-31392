"""
Will prepare image data for object recognition
"""

import cv2
import numpy as np
import glob
import time
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.model_selection import train_test_split

def getImages(filepath):
    images = sorted(glob.glob(filepath + '/*.png'))
    return images

def watchFrames(images, interval, size=None, object=False, rezise=False):

    if not object:
        for i, image in enumerate(images):
            img = cv2.imread(image)
            if rezise:
                img = cv2.resize(img, size)
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, f'Num Frame: {i}', (100, 100), font, 3, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('frame', img)
            cv2.waitKey(interval)
    else:
        for i in range(len(images[0])):
            img = cv2.imread(images[0][i])
            if rezise:
                img = cv2.resize(img, size)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f'Num Frame: {i}', (100, 100), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f'Object: {images[1][i]}', (100, 200), font, 3, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('frame', img)
            cv2.waitKey(interval)


def preproccessData(data, resize, classes, batch=4):
    le = LabelEncoder()
    le.fit(classes)
    labels_encoded = np.array(le.transform(data[1]))
    start = time.time()
    data_points = len(data[0])
    num_batches = math.floor(data_points/batch)
    data_raw = np.zeros((num_batches, batch, 3, resize[0], resize[1]))
    labels_batched_encoded = np.zeros((num_batches, batch))
    j = 0
    for i in range(num_batches):
        for batch_num in range(4):
            img_data = cv2.imread(data[0][j])
            img_data = cv2.resize(img_data, (resize[0], resize[1]))
            img_data = img_data.reshape(3, resize[0], resize[1])
            data_raw[i, batch_num, :, :, :] = img_data
            labels_batched_encoded[i, batch_num] = labels_encoded[j]

            j += 1


    print(f"time it took to load images: {time.time() - start}")

    return data_raw, labels_batched_encoded

def dataLoad(x='./data/imageX.npy', y='./data/imageY.npy', test_size = 0.20):
    X = np.load(x)
    Y = np.load(y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

    return X_train, X_test, y_train, y_test
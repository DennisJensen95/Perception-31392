import numpy as np
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import img_to_array
import torch
import cv2
from PIL import Image
from lib.getDataGoogle import classes_encoder


def process_image(img, dataframe, idx, device, debug=False):
    width, height = img.size
    box = [int(dataframe['XMin'][idx] * width), int(dataframe['YMin'][idx] * height),
           int(dataframe['XMax'][idx] * width), int(dataframe['YMax'][idx] * height)]

    if debug:
        cv_img = np.array(img)
        cv2.rectangle(cv_img, (box[0], box[1]), (box[2], box[3]), color=(255,0,0), thickness=1)
        cv2.imshow('Before crop', cv_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    left = box[0]
    top = box[1]
    right = box[2]
    bottom = box[3]

    img = img.crop((left, top, right, bottom))

    if debug:
        cv_img = np.array(img)
        cv2.imshow('After crop', cv_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    img = img.resize((224, 224))

    if debug:
        cv_img = np.array(img)
        cv2.imshow('After resizing for VGG19 feature extraction', cv_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()


    img = img_to_array(img)
    img = img.reshape((1, img.shape[2], img.shape[0], img.shape[1]))
    img = preprocess_input(img)
    img = torch.from_numpy(np.flip(img, axis=0).copy()).to(device)  # Negative strides

    return img

def extract_features_with_vgg19_cnn(dataframe, featureExtractor, device, save=False, path=None, debug=False):
    # Extract training data
    X = []
    y = []
    for idx in range(len(dataframe['ImgPath'])):
        try:
            img = Image.open(dataframe['ImgPath'][idx])
            img = process_image(img, dataframe, idx, device, debug=debug)
            featureOutput = featureExtractor(img).cpu().detach().numpy().flatten()
            X.append(featureOutput)
            y.append(classes_encoder[dataframe['ClassName'][idx]])
        except RuntimeError as e:
            print(e)
    X = np.asarray(X)
    y = np.asarray(y)

    if save:
        np.savetxt(path + '_X', X, delimiter=',')
        np.savetxt(path + '_y', y, delimiter=',')

    return X, y
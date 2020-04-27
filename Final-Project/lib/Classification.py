import pickle
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import img_to_array
import torch
import numpy as np
from lib.pretrainedNeuralNets import getCNNFeatureExtractVGG19
import cv2
import time

class Classifier:

    def __init__(self, path_clf, path_pca, device):
        self.path_clf = path_clf
        self.path_pca = path_pca
        self.featureExtractor = getCNNFeatureExtractVGG19().to(device)

        self.device = device

        self.load_models()

    def load_models(self):
        self.pca = pickle.load(open(self.path_pca, 'rb'))
        self.clf = pickle.load(open(self.path_clf, 'rb'))

    def process_image(self, img):
        img = img.resize((224, 224))
        img = img_to_array(img)
        img = img.reshape((1, img.shape[2], img.shape[0], img.shape[1]))
        img = preprocess_input(img)
        img = torch.from_numpy(np.flip(img, axis=0).copy()).to(self.device)  # Negative strides
        return img

    def classify_object(self, img):
        features = self.featureExtractor(self.process_image(img)).cpu().detach().numpy().flatten()
        features = features.reshape(1, -1)
        X = self.pca.transform(features)


        class_predicted = self.clf.predict(X)

        return class_predicted

class YOLOClassifier:

    def __init__(self):
        self.yolo = cv2.dnn.readNetFromDarknet('./../darknet/cfg/yolov3.cfg', './../darknet/weights/yolov3.weights')

        self.setup_layers()

    def setup_layers(self):
        ln = self.yolo.getLayerNames()
        self.ln = [ln[i[0] - 1] for i in self.yolo.getUnconnectedOutLayers()]

    def classify(self, img):
        img = np.asarray(img)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        start = time.time()
        layerOutputs = self.yolo.forward()
        end = time.time()
        print(layerOutputs)
        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

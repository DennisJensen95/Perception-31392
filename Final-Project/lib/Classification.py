import pickle
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import img_to_array
import torch
import numpy as np
from lib.pretrainedNeuralNets import getCNNFeatureExtractVGG19
import torchvision.transforms as transforms
import cv2
import time
from lib.NeuralNetClassifier import Net

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

class NeuralNetClassifier:

    def __init__(self):
        self.net_load_path = './data/NeuralNet/Classifier_Model_after_dat.net'
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.classes_encoder = {'Box': 0, 'Book': 1, 'Coffee cup': 2}
        self.classes_decoder = {v: k for k, v in self.classes_encoder.items()}

    def process_image(self, img, device):
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img = self.transform(img).to(device)
        img = img.view(1, 3, 224, 224)

        return img

    def load_model(self, device, eval=True):
        input_shape = (3, 224, 224)
        net = Net(input_shape, classes=3).to(device)
        net.load_state_dict(torch.load(self.net_load_path))
        self.net = net

        if eval:
            self.net.eval()

    def classify_img(self, img, device):
        img = self.process_image(img, device)
        outputs = self.net(img)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy()
        return self.classes_decoder[predicted[0]]


class YOLOClassifier:

    def __init__(self):
        self.yolo = cv2.dnn.readNetFromDarknet('./../darknet/cfg/yolov3.cfg',
                                               './../darknet/weights/yolov3.weights')

        labelsPath = './../darknet/data/9k.names'
        self.LABELS = open(labelsPath).read().strip().split("\n")

        self.setup_layers()

        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
                                   dtype="uint8")

        print(self.LABELS)
        self.class_id_box = self.LABELS.index('box')
        self.class_id_book = self.LABELS.index('book')
        self.class_id_cup = self.LABELS.index('cup')

    def setup_layers(self):
        ln = self.yolo.getLayerNames()
        self.ln = [ln[i[0] - 1] for i in self.yolo.getUnconnectedOutLayers()]

    def classify(self, img):
        img = np.asarray(img)
        (H, W) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        start = time.time()
        layerOutputs = self.yolo.forward(self.ln)
        end = time.time()
        # print(layerOutputs)
        # show timing information on YOLO
        # print("[INFO] YOLO took {:.6f} seconds".format(end - start))
        classIDs = []
        confidences = []
        boxes = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.1:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([int(x), int(y), int(width), int(height)])
                    classIDs.append(classID)
                    confidences.append(float(confidence))
        print(confidences)
        print(boxes)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.3)

        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]],
                                           confidences[i])
                cv2.putText(img, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Image", img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()






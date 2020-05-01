import torch.nn as nn
import pandas as pd
import torch
from torch import optim
import numpy as np
from lib.pretrainedNeuralNets import getCNNFeatureExtractVGG19
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, input_shape, classes, learning_rate=0.001):
        super(Net, self).__init__()

        self.feature = getCNNFeatureExtractVGG19(pretrained=True)

        out_features = self.conv_out_features(input_shape)

        self.classifier = nn.Sequential(
            nn.Linear(out_features, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(2048, classes)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

    def conv_out_features(self, shape):
        o = self.feature(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


def calculateTestAccuracy(loader, classifier, device, classes_decoder):
    correct = 0
    total = 0
    correct_cup = 0
    correct_box = 0
    correct_book = 0
    total_cup = 0
    total_box = 0
    total_book = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()

            for i, label in enumerate(labels):
                if classes_decoder[label] == 'Book':
                    total_book += 1
                    if classes_decoder[predicted[i]] == 'Book':
                        correct_book += 1
                if classes_decoder[label] == 'Coffee cup':
                    total_cup += 1
                    if classes_decoder[predicted[i]] == 'Coffee cup':
                        correct_cup += 1
                if classes_decoder[label] == 'Box':
                    total_box += 1
                    if classes_decoder[predicted[i]] == 'Box':
                        correct_box += 1

    print(f'Accuracy of the network on the {total_cup + total_book + total_box} test images: %d %%' % (
            100 * correct / total))

    print(f'Cup accruacy: {round(correct_cup / total_cup * 100, 2)} % out of {total_cup} cups')
    print(f'Book accruacy: {round(correct_book / total_book * 100, 2)} % out of {total_book} Books')
    print(f'Boxes accruacy: {round(correct_box / total_box * 100, 2)} % out of {total_box} Boxes')

    return 100 * correct / total, correct_box / total_box * 100, \
           correct_book / total_book * 100, correct_cup / total_cup * 100

def plotResultsCsv(csv_file):
    csv_file = pd.read_csv(csv_file)
    plt.figure()
    x = csv_file['ImgNum']*10

    plt.plot(x, csv_file['Loss'])
    plt.plot(x, csv_file['TrainPerc'])
    plt.plot(x, csv_file['TestPerc'])
    plt.legend(['Loss', 'Accuracy train', 'Accuracy test'])
    plt.show()



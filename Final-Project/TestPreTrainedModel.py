import torch
import pandas as pd
from lib.getDataGoogle import SimpleDataLoader
from lib.NeuralNetClassifier import Net, calculateTestAccuracy
import torchvision.transforms as transforms
from lib.NeuralNetClassifier import plotResultsCsv

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
classes_encoder = {'Box': 0, 'Book': 1, 'Coffee cup': 2}
classes_decoder = inv_map = {v: k for k, v in classes_encoder.items()}

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = pd.read_csv('./../pics_/list.csv')
testset = pd.read_csv('./test_csv.csv')

trainLoader = SimpleDataLoader(trainset, transform, classes_encoder)
testLoader = SimpleDataLoader(testset, transform, classes_encoder, base='./Results/Cropped_Images/', suffix='.png')
trainLoader = torch.utils.data.DataLoader(trainLoader, batch_size=10, shuffle=True, num_workers=4)
testLoader = torch.utils.data.DataLoader(testLoader, batch_size=10, shuffle=True, num_workers=4)

image_shape = (3, 224, 224)
num_classes = len(classes_encoder)
classifier = Net(image_shape, num_classes).to(device)

classifier.load_state_dict(torch.load('data/NeuralNet/Classifier_Model_after_dat.net'))
# classifier.load_state_dict(torch.load('data/NeuralNet/Classifier_Model_before_add_dat.net'))

calculateTestAccuracy(trainLoader, classifier, device, classes_decoder)
calculateTestAccuracy(testLoader, classifier, device, classes_decoder)

plotResultsCsv('data/NeuralNet/Classifier_Model_stats_after_dat.csv')
# plotResultsCsv('data/NeuralNet/Classifier_Model_stats_before_add_dat.csv')
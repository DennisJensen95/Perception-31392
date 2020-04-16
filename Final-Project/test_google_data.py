from lib.getDataGoogle import GetGoogleDataset, DataSetLoader, get_transform
from lib.trainResnet50RCNN import getModel
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision
from torchvision.datasets import utils
from lib.visionHelper.utils import collate_fn

classesToSelect = ['Box', 'Book', 'Coffee cup']

get_new_data = False
download = False
prepare_data_set = False

getData = GetGoogleDataset(debug=False, select_classes=classesToSelect)

if get_new_data:
    getData.selectClasses()
    getData.saveData()
else:
    getData.loadData()

if download:
    getData.downloadImages()

print(list(getData.train_annnotations.columns.values))

if prepare_data_set:
    # print(getData.train_annnotations)
    getData.prepareDataSet()

classes_encoder = {'Box': 1, 'Book': 2, 'Coffee cup': 3}

dataframe_train = pd.read_csv('rcnn-test-data/train/train_dataframe.csv')
print(dataframe_train.columns.values)
dataset = DataSetLoader(dataframe_train, get_transform(True), classes_encoder)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=collate_fn
)

num_classes = len(classes_encoder) + 1
model = getModel(num_classes)
images, targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
# print(targets)
# print(images)
output = model(images, targets)   # Returns losses and detections

# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)           # Returns predictions
print(predictions)
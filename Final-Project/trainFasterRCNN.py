from lib.visionHelper.engine import train_one_epoch, evaluate
import lib.visionHelper.utils as utils
from lib.getDataGoogle import DataSetLoader, get_transform, classes_encoder
import torch
import pandas as pd
from lib.trainResnet50RCNN import getModel

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataframe_train = pd.read_csv('rcnn-test-data/train/train_dataframe.csv')
    dataset_train = DataSetLoader(dataframe_train, get_transform(True), classes_encoder)
    dataframe_test = pd.read_csv('rcnn-test-data/test/test_dataframe.csv')
    dataset_test = DataSetLoader(dataframe_test, get_transform(False), classes_encoder)

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    num_classes = len(classes_encoder) + 1
    model = getModel(num_classes)

    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 1

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()






if __name__ == '__main__':
    main()
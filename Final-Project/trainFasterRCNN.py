from lib.visionHelper.engine import train_one_epoch, evaluate
import lib.visionHelper.utils as utils
from lib.getDataGoogle import DataSetLoader, get_transform, classes_encoder
import torch
import pandas as pd
from lib.pretrainedNeuralNets import getModel
import os
import re

save_models = True
pretrain = True
load_model = True

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if save_models:
        model_num = 'model_1'
        save_path = f'./rcnn-test-data/trained_models/{model_num}/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    dataframe_train = pd.read_csv('rcnn-test-data/train/train_dataframe.csv')
    # dataframe_train = dataframe_train.iloc[:100]
    dataset_train = DataSetLoader(dataframe_train, get_transform(True), classes_encoder)
    dataframe_test = pd.read_csv('rcnn-test-data/test/test_dataframe.csv')
    # dataframe_test = dataframe_test.iloc[:20]
    dataset_test = DataSetLoader(dataframe_test, get_transform(False), classes_encoder)

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    num_classes = len(classes_encoder) + 1
    model = getModel(num_classes, pretrain)

    if load_model:
        files = os.listdir(save_path)
        epoch_num = 0
        idx = 0
        if len(files) > 0:
            for i, file in enumerate(files):
                epoch = int(re.findall('epoch_(\d+)', file)[0])
                if epoch_num < epoch:
                    epoch_num = epoch
                    idx = i

            start_epoch = epoch_num
            path_load = save_path + files[idx]
            print(f'Loading state dict from file: {path_load}')
            model.load_state_dict(torch.load(path_load))
        else:
            epoch_num = 0
    else:
        epoch_num = 0

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
    if load_model:
        num_epochs = 150 - epoch_num
    else:
        num_epochs = 150

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=100)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

        if save_models:
            path_to_save = save_path + f'{model_num}_epoch_{epoch + epoch_num}'
            torch.save(model.state_dict(), path_to_save)

if __name__ == '__main__':
    main()
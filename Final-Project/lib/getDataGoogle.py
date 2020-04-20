# Data from https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=segmentation&r=false&c=%2Fm%2F0bt_c3
# Extract Box, Book, Coffee Cup

# Guide https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import pandas as pd
import numpy as np
import random
import os
import shutil
from skimage import io
import re
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from lib.visionHelper import transforms as T
import cv2

class GetGoogleDataset(object):

    def __init__(self, debug, select_classes, num_sample_select=1000):
        self.imgs = None
        self.sub_selection_num = num_sample_select
        self.classes = select_classes
        self.save_test_data = 'rcnn-test-data/sub_sampled/'
        self.save_images_base = 'rcnn-test-data/'
        class_descriptor_fn = './rcnn-test-data/class-descriptions-boxable.csv'
        train_annotations_fn = './rcnn-test-data/oidv6-train-annotations-bbox.csv'
        train_images_fn = './rcnn-test-data/train-images-boxable.csv'

        self.class_descriptor = pd.read_csv(class_descriptor_fn)
        self.train_annnotations = pd.read_csv(train_annotations_fn)
        self.train_images = pd.read_csv(train_images_fn)

        self.sub_sample_data = {}
        self.sub_sample_img_url = {}

        self.annotations_data_frames = {}
        self.label_names = []
        for _class in self.classes:
            class_pd = self.class_descriptor[self.class_descriptor['class'] == _class]
            # Get label for specific class
            label = class_pd['name'].values[0]
            self.label_names.append(label)
            # Box drawing around images
            annotations = self.train_annnotations[self.train_annnotations['LabelName'] == label]
            self.annotations_data_frames.update({_class: annotations})

        self.loaded_data = False

        self.debug = debug
        if debug:
            print("Class descriptor headers")
            print(self.class_descriptor.head())

            print("Train annotations headers")
            print(self.train_annnotations.head())

            print("Train images headers")
            print(self.train_images.head())

    def selectClasses(self):
        # Find label name for the classes selected
        self.label_names = []
        class_data_frames = []
        self.annotations_data_frames = {}
        self.sub_sample_data = {}
        self.sub_sample_img_url = {}
        for _class in self.classes:
            # Get class pandas dataframe
            class_pd = self.class_descriptor[self.class_descriptor['class'] == _class]
            class_data_frames.append(class_pd)
            # Get label for specific class
            label = class_pd['name'].values[0]
            self.label_names.append(label)

            # Box drawing around images
            annotations = self.train_annnotations[self.train_annnotations['LabelName'] == label]
            self.annotations_data_frames.update({_class: annotations})

            # images with unique items in them
            contains_object_id = np.unique(annotations['ImageID'])
            copy_object_ids = contains_object_id.copy()
            random.seed(5)
            random.shuffle(copy_object_ids)

            # Sub Sampling
            sub_data_sample = copy_object_ids
            self.sub_sample_data.update({_class: sub_data_sample})

            sub_sample_img_url = [self.train_images[self.train_images['image_name'] == name + '.jpg'] for name in
                                  sub_data_sample]
            self.sub_sample_img_url.update({_class: sub_sample_img_url})

            if self.debug:
                print('\n')
                print(f'\t {class_pd}')
                print("\n")

                print(f'There are {len(annotations)} {_class}s in the dataset.')

                print(f'There are {len(contains_object_id)} images with more or one {_class} in the image.')

        self.loaded_data = True

    def saveData(self):
        """
        Save sampled data
        """
        for _class in self.classes:
            pd_data_frame = pd.DataFrame()
            for i, sub_sample in enumerate(self.sub_sample_img_url[_class]):
                pd_data_frame = pd_data_frame.append(self.sub_sample_img_url[_class][i], ignore_index=True)

            pd_data_frame.to_csv(os.path.join(self.save_test_data, f'{_class}_img_url.csv'))

    def loadData(self):
        """
        Load saved data
        """
        for _class in self.classes:
            data = pd.read_csv(os.path.join(self.save_test_data, f'{_class}_img_url.csv'))
            self.sub_sample_img_url.update({_class: data})

    def downloadImages(self, del_folders=False):

        for _class in self.classes:
            path = os.path.join(self.save_images_base, _class)

            if os.path.exists(path) and del_folders:
                shutil.rmtree(path)
                os.mkdir(path)
            elif os.path.exists(path) and not del_folders:
                continue
            elif not os.path.exists(path):
                os.mkdir(path)

            for url in self.sub_sample_img_url[_class]['image_url']:
                # print(url)
                img = io.imread(url)
                name = re.findall('/(\w+.jpg)', url)[0]
                saved_path = os.path.join(path, name)
                io.imsave(saved_path, img)


    def prepareDataSet(self):
        path_test = os.path.join(self.save_images_base, 'test')
        path_train = os.path.join(self.save_images_base, 'train')

        train_df = pd.DataFrame(columns=['ImgPath', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])
        test_df = pd.DataFrame(columns=['File', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])


        for _class in self.classes:
            print(f'Making training set for class {_class} \n')

            path_images = os.path.join(self.save_images_base, _class)
            images = os.listdir(path_images)

            X_train, X_test = train_test_split(images, test_size=0.20)

            print(f'Doing train')
            print(f'len train {len(X_train)}')
            for i in range(len(X_train)):
                img_name = X_train[i]
                if i % 100 == 0:
                    print(f'Images left to sort: {len(X_train) - i}')
                img_id = re.findall('(\w+).jpg', img_name)[0]
                data_img = self.annotations_data_frames[_class][self.annotations_data_frames[_class]['ImageID'] == img_id]
                img_path = os.path.join(path_images, img_name)
                for idx, row in data_img.iterrows():
                    label = row['LabelName']
                    for j in range(len(self.label_names)):
                        if label == self.label_names[j]:
                            train_df = train_df.append({'ImgPath': img_path,
                                                        'XMin': row['XMin'],
                                                        'XMax': row['XMax'],
                                                        'YMin': row['YMin'],
                                                        'YMax': row['YMax'],
                                                        'ClassName': self.classes[j]},
                                                       ignore_index=True)

            print(f'Doing test')
            print(f'len test: {len(X_test)}')
            for i in range(len(X_test)):
                img_name = X_test[i]
                img_id = re.findall('(\w+).jpg', img_name)[0]
                data_img = self.annotations_data_frames[_class][self.annotations_data_frames[_class]['ImageID'] == img_id]
                img_path = os.path.join(path_images, img_name)

                if i % 100 == 0:
                    print(f'Images left to sort: {len(X_test) - i}')

                for idx, row in data_img.iterrows():
                    label = row['LabelName']
                    for j in range(len(self.label_names)):
                        if label == self.label_names[j]:
                            test_df = test_df.append({'ImgPath': img_path,
                                                      'XMin': row['XMin'],
                                                      'XMax': row['XMax'],
                                                      'YMin': row['YMin'],
                                                      'YMax': row['YMax'],
                                                      'ClassName': self.classes[j]},
                                                     ignore_index=True)

        train_df.to_csv(os.path.join(path_train, 'train_dataframe.csv'))
        test_df.to_csv(os.path.join(path_test, 'test_dataframe.csv'))


class DataSetLoader(object):

    def __init__(self, dataframe, transforms, classes_encoder):
        self.transforms = transforms
        self.dataframe = dataframe
        self.classes_encoder = classes_encoder


    def __getitem__(self, idx):
        """Get an image"""
        img = Image.open(self.dataframe['ImgPath'][idx]).convert('RGB')
        height, width, channels = np.asarray(img).shape

        # print(f'width: {width}, height: {height}')

        # img = cv2.imread(self.dataframe['ImgPath'][idx])

        box = [[int(self.dataframe['XMin'][idx] * width), int(self.dataframe['YMin'][idx] * height),
               int(self.dataframe['XMax'][idx] * width), int(self.dataframe['YMax'][idx] * height)]]

        # cv2.rectangle(img, (box[0][0], box[0][1]), (box[0][2], box[0][3]), color=(255,0,0), thickness=1)
        # cv2.imshow('target', img)
        # cv2.waitKey(0)

        box = torch.as_tensor(box, dtype=torch.float32)
        area = (box[:, 3] - box[:, 1]) * (box[:, 2] - box[:, 0])
        img_id = torch.tensor([idx])

        iscrowd = torch.zeros((1,), dtype=torch.int64)

        # Class labels encoded in numbers
        labels = [self.classes_encoder[self.dataframe['ClassName'][idx]]]
        labels = torch.as_tensor(labels, dtype=torch.int64)


        target = {}
        target["boxes"] = box
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.dataframe['ImgPath'])

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

classes_encoder = {'Box': 1, 'Book': 2, 'Coffee cup': 3}












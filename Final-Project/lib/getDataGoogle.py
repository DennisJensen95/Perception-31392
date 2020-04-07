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
        labels = []
        class_data_frames = []
        annotations_data_frames = []
        self.sub_sample_data = {}
        self.sub_sample_img_url = {}
        for _class in self.classes:
            # Get class pandas dataframe
            class_pd = self.class_descriptor[self.class_descriptor['class'] == _class]
            class_data_frames.append(class_pd)
            # Get label for specific class
            label = class_pd['name'].values[0]
            labels.append(label)

            # Box drawing around images
            annotations = self.train_annnotations[self.train_annnotations['LabelName'] == label]
            annotations_data_frames.append(annotations)

            # images with unique items in them
            contains_object_id = np.unique(annotations['ImageID'])
            copy_object_ids = contains_object_id.copy()
            random.seed(5)
            random.shuffle(copy_object_ids)

            # Sub Sampling
            sub_data_sample = copy_object_ids[:self.sub_selection_num]
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










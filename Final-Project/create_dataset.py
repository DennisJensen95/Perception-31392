import numpy as np
import pandas as pd
from PIL import Image
import cv2
import click
from lib.getDataGoogle import classes_encoder

def search_through_images(dataframe):

    img = Image.open(dataframe['ImgPath'])
    width, height = img.size
    box = [int(dataframe['XMin'] * width), int(dataframe['YMin'] * height),
           int(dataframe['XMax'] * width), int(dataframe['YMax'] * height)]

    left = box[0]
    top = box[1]
    right = box[2]
    bottom = box[3]

    img = img.crop((left, top, right, bottom))
    print(dataframe['ClassName'])
    cv_img = np.array(img)
    cv2.imshow('Testing Images', cv_img)
    cv2.waitKey(100)

    if click.confirm('Add image?', default=True):
        cv2.destroyAllWindows()
        return True
    else:
        cv2.destroyAllWindows()
        return False


def main():

    dataframe_train = pd.read_csv('rcnn-test-data/train/train_dataframe.csv').sample(frac=1).reset_index(drop=True)
    dataframe_test = pd.read_csv('rcnn-test-data/test/test_dataframe.csv').sample(frac=1).reset_index(drop=True)

    new_dataframe_train = pd.read_csv('self_made_train.csv')
    new_dataframe_test = pd.read_csv('self_made_test.csv')

    books = 0
    boxes = 0
    cups = 0
    for idx in range(len(dataframe_train['ImgPath'])):
        label = dataframe_train['ClassName'][idx]

        if label == 'Box' and boxes >= 30:
            continue
        elif label == 'Coffee cup' and cups >= 30:
            continue
        elif label == 'Book' and books >= 30:
            continue

        add = search_through_images(dataframe_train.to_dict(orient='records')[idx])

        if add:
            new_dataframe_train = new_dataframe_train.append(dataframe_train.iloc[idx, :], ignore_index=True)
            if label == 'Box':
                boxes += 1
            elif label == 'Coffee cup':
                cups += 1
            elif label == 'Book':
                books += 1

    books = 0
    boxes = 0
    cups = 0
    for idx in range(len(dataframe_test['ImgPath'])):
        label = dataframe_test['ClassName'][idx]

        if label == 'Box' and boxes >= 5:
            continue
        elif label == 'Coffee cup' and cups >= 5:
            continue
        elif label == 'Book' and books >= 5:
            continue

        add = search_through_images(dataframe_test.to_dict(orient='records')[idx])

        if add:
            new_dataframe_test = new_dataframe_test.append(dataframe_test.iloc[idx, :], ignore_index=True)
            if label == 'Box':
                boxes += 1
            elif label == 'Coffee cup':
                cups += 1
            elif label == 'Book':
                books += 1

    test_data = new_dataframe_test.copy()
    train_data = new_dataframe_train.copy()
    train_data.drop_duplicates().to_csv('self_made_train.csv')
    test_data.drop_duplicates().to_csv('self_made_test.csv')

if __name__ == '__main__':
    main()
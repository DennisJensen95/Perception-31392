from lib.fastRCNNPretrained import *
from lib.prepareData import *
from lib.pretrainedNeuralNets import getModel
from lib.getDataGoogle import classes_encoder
from torchvision import transforms, models
import torch
import glob
import pandas as pd

test_1 = True

if test_1:

    images = sorted(glob.glob('data/Stereo_conveyor_without_occlusions/left/*.png'))

    image = images[1330]
    debug = False
    img = cv2.imread(image)
    if debug:
        cv2.imshow('test', img)
        cv2.waitKey(500)

    # object_detection_api(image, 50, threshold=0.7)

    images = getImages('data/Stereo_conveyor_without_occlusions/left/')

    box_1 = [100, 250, 'Box']
    box_2 = [330, 450, 'Box']
    book_1 = [500, 630, 'Book']
    book_2 = [660, 830, 'Book']
    book_3 = [870, 1030, 'Book']
    cup_1 = [1070, 1220, 'Coffee cup']
    cup_2 = [1260, 1460, 'Coffee cup']

    intervals = [box_1, box_2, book_1, book_2, book_3, cup_1, cup_2]
    y = np.empty(0)
    x = np.empty(0)
    for i in range(len(intervals)):
        object_images = np.asarray(images[intervals[i][0]:intervals[i][1]])
        frames = intervals[i][1] - intervals[i][0]
        y_objects = np.full(frames, intervals[i][2])
        y = np.append(y, y_objects)
        x = np.append(x, object_images)


    """ Test trained model """

    images = sorted(glob.glob('data/Stereo_conveyor_without_occlusions/left/*.png'))
    image_file = images[400]

    our_data = np.array([x, y]).T
    books = our_data[y == 'Book'][:10]
    cups = our_data[y == 'Coffee cup'][:10]
    boxes = our_data[y == 'Box'][:10]
    images = []
    images.extend(books)
    images.extend(cups)
    images.extend(boxes)

    training_images = pd.read_csv('rcnn-test-data/train/train_dataframe.csv')

    data_train = np.array([training_images['ImgPath'], training_images['ClassName']]).T
    books_train = data_train[training_images['ClassName'] == 'Book'][:10]
    cups_train = data_train[training_images['ClassName'] == 'Coffee cup'][:10]
    boxes_train = data_train[training_images['ClassName'] == 'Box'][:10]

    images_train = []
    images_train.extend(books_train)
    images_train.extend(cups_train)
    images_train.extend(boxes_train)

    # Get a decoder
    classes_decoder = inv_map = {v: k for k, v in classes_encoder.items()}

    num_classes = len(classes_encoder) + 1
    model = getModel(num_classes)
    model.load_state_dict(torch.load('rcnn-test-data/trained_models/model_1/model_1_epoch_99'))
    # model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    test_training_images = False
    if test_training_images:
        images = images_train

    print(images)

    for i in range(len(images)):
        image_file = images[i][0]
        img = Image.open(image_file).convert('RGB')
        img = transforms.ToTensor()(img)
        output = model([img])


        img = cv2.imread(image_file)
        boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(output[0]['boxes'].detach().numpy())]
        scores = output[0]['scores'].detach().numpy()
        labels = output[0]['labels'].numpy()

        true_label = images[i][1]
        # best_pred_true_label = np.where(labels == classes_encoder[true_label])

        for i, box in enumerate(boxes):
            if i > 1:
                break
            cv2.rectangle(img, box[0], box[1], color=(0,255,0), thickness=1)
            cv2.putText(img, f'Class: {classes_decoder[labels[i]]} score: {round(scores[i] * 100)} %', box[0],  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=1)
            print(f'Class: {classes_decoder[labels[i]]} score: {round(scores[i] * 100)} %')


        cv2.imshow('results', img)
        cv2.waitKey(100)
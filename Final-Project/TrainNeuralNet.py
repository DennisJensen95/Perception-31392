import torch
import pandas as pd
from lib.getDataGoogle import SimpleDataLoader, classes_encoder
from lib.NeuralNetClassifier import Net, calculateTestAccuracy
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import time
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
classes_encoder = {'Box': 0, 'Book': 1, 'Coffee cup': 2}

classes_decoder = inv_map = {v: k for k, v in classes_encoder.items()}

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = pd.read_csv('./../pics_/list.csv')
testset = pd.read_csv('./test_csv.csv')

trainLoader = SimpleDataLoader(trainset, transform, classes_encoder)
testLoader = SimpleDataLoader(testset, transform, classes_encoder, base='./Results/Cropped_Images/', suffix='.png')
trainLoader = torch.utils.data.DataLoader(trainLoader, batch_size=50, shuffle=True, num_workers=4)
testLoader = torch.utils.data.DataLoader(testLoader, batch_size=50, shuffle=True, num_workers=4)

lr = 0.00001
image_shape = (3, 224, 224)
num_classes = len(classes_encoder)
classifier = Net(image_shape, num_classes, lr).to(device)
epochs = 5

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainLoader)
images, labels = dataiter.next()

path_to_save = 'data/NeuralNet/Classifier_Model.net'

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
print(labels)
print(labels.numpy()[0])
print(' '.join('%5s' % classes_decoder[labels.numpy()[j]] for j in range(1)))
start = time.time()
loss_vec = []
train_perc = []
test_perc = []
box_perc_train = []
book_perc_train = []
cup_perc_train = []
box_perc_test = []
book_perc_test = []
cup_perc_test = []
img_trained = []
img_num = 0
highest_perc = 0
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        img_num += 1
        img, labels = data

        img = img.to(device)
        labels = labels.to(device)
        # print(img.shape)
        classifier.optimizer.zero_grad()

        outputs = classifier(img)
        loss = classifier.criterion(outputs, labels)
        loss.backward()
        classifier.optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 60 == 0:
            # score_test, box_perc, book_perc, cup_perc = calculateTestAccuracy(testLoader, classifier, device, classes_decoder)
            # test_perc.append(score_test)
            # cup_perc_test.append(cup_perc)
            # book_perc_test.append(book_perc)
            # box_perc_test.append(box_perc)
            #
            # score, box_perc, book_perc, cup_perc = calculateTestAccuracy(trainLoader, classifier, device, classes_decoder)
            # train_perc.append(score)
            # cup_perc_train.append(cup_perc)
            # book_perc_train.append(book_perc)
            # box_perc_train.append(box_perc)

            loss_vec.append(running_loss)

            # if highest_perc < score_test:
            #     highest_perc = score_test
            #     torch.save(classifier.state_dict(), path_to_save)


            img_trained.append(img_num)

            print(f'Epoch: {epoch}: loss {running_loss}')
        running_loss = 0.0

    print('Finished Training')

perc_test, _, _, _ = calculateTestAccuracy(testLoader, classifier, device, classes_decoder)

calculateTestAccuracy(trainLoader, classifier, device, classes_decoder)

print(f'Time it took {(time.time() - start)/60}')
df_dat = pd.DataFrame(list(zip(loss_vec, train_perc, test_perc, cup_perc_train,
                               box_perc_train, book_perc_train, cup_perc_test,
                               box_perc_test, book_perc_test,
                               img_trained)), columns=['Loss', 'TrainPerc','TestPerc',
                                                       'TrainCupPerc', 'TrainBoxPerc', 'TrainBookPerc',
                                                       'TestCupPerc', 'TestBoxPerc', 'TestBookPerc','ImgNum'])

path_to_save_results = 'data/NeuralNet/Classifier_Model_stats.csv'

df_dat.to_csv(path_to_save_results)
if perc_test > highest_perc:
    torch.save(classifier.state_dict(), path_to_save)
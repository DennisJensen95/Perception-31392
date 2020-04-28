import torch
import pandas as pd
from lib.getDataGoogle import SimpleDataLoader, classes_encoder
from lib.NeuralNetClassifier import Net
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

classes_encoder = {'Box': 0, 'Book': 1, 'Coffee cup': 2}

classes_decoder = inv_map = {v: k for k, v in classes_encoder.items()}

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5), (0.5, 0.5))])

trainset = pd.read_csv('./../pics_/list.csv')
testset = pd.read_csv('./test_csv.csv')

trainLoader = SimpleDataLoader(trainset, transform, classes_encoder)
testLoader = SimpleDataLoader(testset, transform, classes_encoder, base='./Results/Cropped_Images/', suffix='.png')
trainLoader = torch.utils.data.DataLoader(trainLoader, batch_size=10, shuffle=True, num_workers=4)
testLoader = torch.utils.data.DataLoader(testLoader, batch_size=10, shuffle=True, num_workers=4)

lr = 0.0001
image_shape = (2, 64, 64)
num_classes = len(classes_encoder)
classifier = Net(image_shape, num_classes, lr).to(device)
epochs = 20

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainLoader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
print(labels)
print(labels.numpy()[0])
print(' '.join('%5s' % classes_decoder[labels.numpy()[j]] for j in range(1)))

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        img, labels = data

        img = img.to(device)
        labels = labels.to(device)

        classifier.optimizer.zero_grad()

        outputs = classifier(img)
        loss = classifier.criterion(outputs, labels)
        loss.backward()
        classifier.optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 0:
            print(f'Epoch: {epoch}: loss {running_loss}')
        running_loss = 0.0

    print('Finished Training')

correct = 0
total = 0
correct_cup = 0
correct_box = 0
correct_book = 0
total_cup = 0
total_box = 0
total_book = 0
with torch.no_grad():
    for data in testLoader:
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

print('Accuracy of the network on the 1300 test images: %d %%' % (
    100 * correct / total))

print(f'Cup accruacy: {round(correct_cup/total_cup * 100,2)} % out of {total_cup} cups')
print(f'Book accruacy: {round(correct_book/total_book * 100, 2)} % out of {total_book} Books')
print(f'Boxes accruacy: {round(correct_box/total_box * 100, 2)} % out of {total_box} Boxes')

correct = 0
total = 0
correct_cup = 0
correct_box = 0
correct_book = 0
total_cup = 0
total_box = 0
total_book = 0
with torch.no_grad():
    for data in trainLoader:
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


print('Accuracy of the network on the 193 traning images: %d %%' % (
    100 * correct / total))

print(f'Cup accruacy: {round(correct_cup/total_cup * 100,2)} % out of {total_cup} cups')
print(f'Book accruacy: {round(correct_book/total_book * 100, 2)} % out of {total_book} Books')
print(f'Boxes accruacy: {round(correct_box/total_box * 100, 2)} % out of {total_box} Boxes')

path_to_save = './data/NeuralNet/Classifier_Model.net'
torch.save(classifier.state_dict(), path_to_save)
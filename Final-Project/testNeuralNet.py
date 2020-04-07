import torch
from lib.prepareData import *
from torch.autograd import Variable

net = torch.load('./data/NeuralNet/testModel.net')

images = getImages('data/stereo_conveyor_without_occlusions/left/')

box_1 = [100, 250, 'box']
box_2 = [330, 450, 'box']
book_1 = [500, 630, 'book']
book_2 = [660, 830, 'book']
book_3 = [870, 1030, 'book']
cup_1 = [1070, 1220, 'cup']
cup_2 = [1260, 1460, 'cup']

intervals = [box_1, box_2, book_1, book_2, book_3, cup_1, cup_2]
y = np.empty(0)
x = np.empty(0)
for i in range(len(intervals)):
    object_images = np.asarray(images[intervals[i][0]:intervals[i][1]])
    frames = intervals[i][1] - intervals[i][0]
    y_objects = np.full(frames, intervals[i][2])
    y = np.append(y, y_objects)
    x = np.append(x, object_images)
data = [x, y]


test_img = cv2.imread(data[0][800])
shape = test_img.shape
scale = 10
width, height = int(shape[1]/scale), int(shape[0]/scale)
test_img = cv2.resize(test_img, (width, height))
test_img = test_img.reshape(1, 3, width, height)
inputs = torch.from_numpy(test_img).float()
inputs = Variable(inputs).to('cuda:0')

output = net(inputs)
print(output)

correct = 0
total = 0

gpu_run = True
device='cuda:0'

X_train, X_test, y_train, y_test = dataLoad()

for i in range(len(y_train)):
    inputs = torch.from_numpy(X_train[i]).float()
    labels = torch.from_numpy(y_train[i]).float()
    if gpu_run:
        labels = labels.to(device)
        inputs = Variable(inputs).to(device)
    else:
        inputs = Variable(inputs)

    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the {} test images: {:4.2f} %'.format(
    len(y_train), 100 * correct / total))
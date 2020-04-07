import torch.optim as optim
from lib.NeuralNetwork import *
import torch
from torch.autograd import Variable
from lib.prepareData import *

X_train, X_test, y_train, y_test = dataLoad()

run = 'gpu'

if run == 'cpu':
    gpu_run = False
else:
    gpu_run = True

labels = ['box', 'book', 'cup']

net = Net(len(labels))

if torch.cuda.is_available() and gpu_run:
    print("Moving the network to my GPU.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
else:
    device = None

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.6)

num_epoch = 3  # Your code here!

net.train()
for epoch in range(num_epoch):  # loop over the dataset multiple times
    print(f'Epoch: {epoch}')
    running_loss = 0.0
    for i in range(len(y_train)):
        # get the inputs
        inputs = torch.from_numpy(X_train[i]).float()
        labels = torch.from_numpy(y_train[i]).float()
        # wrap them in Variable
        if gpu_run:
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device, dtype=torch.long)
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()
        # print(f'outputs size: {outputs}\nlabels size: {labels}')

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 0:  # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')

torch.save(net, './data/NeuralNet/testModel.net')
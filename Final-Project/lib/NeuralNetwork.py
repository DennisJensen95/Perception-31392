import torch.nn as nn


input_channel = 3
width_input = 128
height_input = 72

# First convolutional layer
num_filters_conv1 = 16
kernel_size_conv1 = 5  # [kernel_height, kernel_width]
stride_conv1 = 1  # [stride_height, stride_width]
padding_conv1 = 2

# First max pooling
kernel_size_pool1 = 2
stride_pool1 = 2
padding_pool1 = 1

# Second Convolutional layer
num_filters_conv2 = 16
kernel_size_conv2 = 5  # [kernel_height, kernel_width]
stride_conv2 = 1  # [stride_height, stride_width]
padding_conv2 = 2



class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.activation = nn.ReLU()
        self.cost = nn.CrossEntropyLoss()
        self.drop = nn.Dropout2d(p=0.4)

        # First layer
        self.conv1 = nn.Conv2d(in_channels=input_channel,
                               out_channels=num_filters_conv1,
                               kernel_size=kernel_size_conv1,
                               stride=stride_conv1,
                               padding=padding_conv1)

        # Dimensions of output of first layer
        output_height_conv1 = self.out_dim(height_input, kernel_size_conv1, padding_conv1, stride_conv1)
        output_width_conv1 = self.out_dim(width_input, kernel_size_conv1, padding_conv1, stride_conv1)

        self.output_conv1 = num_filters_conv1 * output_height_conv1 * output_width_conv1

        # Max pooling first convolutional layer
        self.pool1 = nn.MaxPool2d(kernel_size=kernel_size_pool1, stride=stride_pool1, padding=padding_pool1)

        output_height_pool1 = self.out_dim(output_height_conv1, kernel_size_pool1, padding_pool1, stride_pool1)
        output_width_pool1 = self.out_dim(output_width_conv1, kernel_size_pool1, padding_pool1, stride_pool1)

        self.pool1_output = num_filters_conv1 * output_height_pool1 * output_width_pool1

        # Second Convolutional layer
        self.conv2 = nn.Conv2d(in_channels=num_filters_conv1,
                               out_channels=num_filters_conv2,
                               kernel_size=kernel_size_conv2,
                               stride=stride_conv2,
                               padding=padding_conv2)

        output_height_conv2 = self.out_dim(output_height_pool1, kernel_size_conv2, padding_conv2, stride_conv2)
        output_width_conv2 = self.out_dim(output_width_pool1, kernel_size_conv2, padding_conv2, stride_conv2)

        self.output_conv2 = num_filters_conv2 * output_height_conv2 * output_width_conv2

        self.dense_1_input = self.output_conv2

        # First Linear dense layer with output features
        self.dense_1 = nn.Linear(in_features=self.dense_1_input,
                                 out_features=self.num_classes,
                                 bias=True)

        # Output layer, as many outputs as their are classes to classify
        self.output_layer = nn.Linear(in_features=self.num_classes,
                                      out_features=self.num_classes,
                                      bias=False)

    def forward(self, x):
        # First layer convolutional layer
        x = self.conv1(x)
        x = self.activation(x)

        # Max pool
        x = self.pool1(x)

        # Second convolutional
        x = self.conv2(x)

        # Format x input to a dense layer
        x = x.view(-1, self.dense_1_input)

        # Second layer dense layer
        x = self.dense_1(x)
        x = self.activation(x)
        # x = self.drop(x)

        # Output layer
        x = self.output_layer(x)
        return x

    def out_dim(self, orig_dim, kernel_size, padding, stride):
        return int((orig_dim - kernel_size + 2 * padding) / stride + 1)
import torch.nn as nn


class FConvNet(nn.Module):
    def __init__(self):
        super(FConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=(4, 0), padding_mode='circular')
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=(4, 0), padding_mode='circular')
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=(4, 0), padding_mode='circular')

        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=(4, 0), padding_mode='circular')
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=(4, 0), padding_mode='circular')
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=(4, 0), padding_mode='circular')

        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), padding=(0, 1))

        self.conv7 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=(4, 0), padding_mode='circular')
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=(4, 0), padding_mode='circular')
        self.conv9 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=(4, 0), padding_mode='circular')

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=(4, 0), padding_mode='circular')
        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=(4, 0), padding_mode='circular')

        self.conv12 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)

        self.relu = nn.ReLU()

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.00001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.pool1(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)

        x = self.pool2(x)

        x = self.conv7(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.relu(x)

        x = self.pool3(x)

        x = self.conv10(x)
        x = self.relu(x)
        x = self.conv11(x)
        x = self.relu(x)
        x = self.conv12(x)

        return x.view(-1, 2048)

import torch
from torch import nn


class VGGnet(nn.Module):
    # input 48 * 48 * 1
    def __init__(self):
        super(VGGnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
            # out 48 * 48 *32
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)  # out 24 * 24 * 64
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)  # out 12 * 12 * 64
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(12 * 12 * 64, 2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.Dropout(0.5)
        )

        self.fc1 = nn.Linear(12 * 12 * 64, 1 * 1 * 2048)
        self.dp1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1 * 1 * 2048, 1 * 1 * 1024)
        self.dp2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1 * 1 * 1024, 1 * 1 * 8)

    def forward(self, x):
        # print("~~~~~~~~~")
        y = self.conv1(x)
        # print(y.shape)  # After conv1
        # print('------------')

        y = self.conv2(y)
        # print(y.shape)  # After conv2
        # print('------------')

        y = self.conv3(y)
        # print(y.shape)  # After conv3
        # print('------------')

        # Check the shape of y before flattening (should be [batch_size, 64, 12, 12])
        # print("Shape before flattening:", y.shape)

        # Flattening the output of conv3 before feeding it to fully connected layers
        y = y.view(y.size(0),
                   -1)  # Or use torch.flatten(y, 1) to flatten the tensor (this ensures the shape is [batch_size, 9216])

        # print("Shape after flattening:", y.shape)  # Ensure this is [batch_size, 9216]

        # Pass through fully connected layers
        y = self.fc1(y)
        y = self.dp1(y)
        y = self.fc2(y)
        y = self.dp2(y)
        y = self.fc3(y)


        # Category prediction (assuming binary classification)
        # category = torch.sigmoid(y[:, 0:1])
        # return category
        # return torch.softmax(y, dim=-1)
        return y


# test
if __name__ == "__main__":
    model = VGGnet()

    input = torch.randn(64, 1, 48, 48)

    out = model(input)

    print('!~~~~~~~~~~~~~~~!')
    print(out.shape)
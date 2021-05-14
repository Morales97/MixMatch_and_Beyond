import torch
import torch.nn as nn
from torchsummary import summary
import math

class BasicBlock(nn.Module):

    def __init__(self, n_features):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_features)   # when trying EMA with many epochs, try using BN with momentum=0.001
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=(3, 3), padding=1)

        self.bn2 = nn.BatchNorm2d(n_features)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x = self.relu(self.bn1(x))
        identity = x
        out = self.conv1(x)

        out = self.relu(self.bn2(out))
        out = self.conv2(out)

        out += identity
        return out


class TransitionBlock(nn.Module):
    """
    Block used for one or both of the following reasons:
        - Downsample input by 2 using conv of stride=2
        - Adapt number of features using 1x1 filters
    """
    def __init__(self, in_f, out_f, downsample):

        if downsample:
            stride = 2
        else:
            stride = 1

        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_f)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv1 = nn.Conv2d(in_f, out_f, kernel_size=(3, 3), padding=1, stride=stride)

        self.bn2 = nn.BatchNorm2d(out_f)
        self.conv2 = nn.Conv2d(out_f, out_f, kernel_size=(3, 3), padding=1)

        # Shortcut connection for identity to match dimensions
        self.shortcut = nn.Conv2d(in_f, out_f, kernel_size=(1, 1), stride=stride)

    def forward(self, x):
        x = self.relu(self.bn1(x))
        identity = self.shortcut(x)
        out = self.conv1(x)

        out = self.relu(self.bn2(out))
        out = self.conv2(out)

        out += identity
        return out


class ConvGroup(nn.Module):

    def __init__(self, in_features, out_features, blocks, downsample=True):
        super(ConvGroup, self).__init__()

        self.conv_blocks = nn.Sequential(TransitionBlock(in_features, out_features, downsample),
                                         *[BasicBlock(out_features) for _ in range(blocks - 1)])

    def forward(self, x):
        return self.conv_blocks(x)


class WideResNet(nn.Module):

    def __init__(self, depth, k, n_out):
        super(WideResNet, self).__init__()

        assert (depth - 4) % 6 == 0, "depth must be 6*n + 4"
        n = int((depth - 4) / 6)
        n_features = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, n_features[0], kernel_size=(3, 3), padding=1)
        self.conv_group1 = ConvGroup(n_features[0], n_features[1], blocks=n, downsample=False)
        self.conv_group2 = ConvGroup(n_features[1], n_features[2], blocks=n)
        self.conv_group3 = ConvGroup(n_features[2], n_features[3], blocks=n)
        self.bn = nn.BatchNorm2d(n_features[3])
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.linear = nn.Linear(n_features[3], n_out)

        """
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
        """

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_group1(x)
        x = self.conv_group2(x)
        x = self.conv_group3(x)
        x = self.relu(self.bn(x))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


if __name__ == '__main__':

    model = WideResNet(depth=28, k=2, n_out=10)
    summary(model, (3, 32, 32))

import torch
import torch.nn as nn
from typing import Type

class SkipBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, stride=1, 
                dilation=1, padding=1, skip=True, downsample_block=None):
        super(SkipBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.downsample_block = downsample_block
        self.skip = skip

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding =self.padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=1, padding =self.padding, bias=False)
        
        

    def forward(self, x):
        x_identity = x
        x = self.conv1(x)
        x = self.bn(x)
        if self.skip:
            x = self.conv2(x)
            x = self.bn(x)
            if self.downsample_block is not None:
                x_identity = self.downsample_block(x_identity)
            x += x_identity
        out = self.relu(x)

        return out

    
class ResNet(nn.Module):
    def __init__(self,  block: Type[SkipBlock], n_channels = 3, n_layers = 5, n_classes = 1000):
        super(ResNet, self).__init__()
        self.in_channels = n_channels
        self.out_channels = 64
        self.layer0 = SkipBlock(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size = 7, padding=3, 
                            stride=2, skip=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = [2, 2, 2, 2]
        self.layer1 = self.make_layer_(block, out_channels = 64, stride = 1, n_blocks=self.layers[0])
        self.layer2 = self.make_layer_(block, out_channels = 128, stride = 2, n_blocks=self.layers[1])
        self.layer3 = self.make_layer_(block, out_channels = 256, stride = 2, n_blocks=self.layers[2])
        self.layer4 = self.make_layer_(block, out_channels = 512, stride = 2, n_blocks=self.layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_classes)

    def make_layer_(self, block: Type[SkipBlock], out_channels, stride = 1, n_blocks = 5):
        downsample_block = None
        if stride != 1:
            downsample_block = nn.Sequential(
                nn.Conv2d(
                    self.out_channels, 
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(
            block(
                in_channels=self.out_channels, out_channels=out_channels, stride=stride, downsample_block=downsample_block
            )
        )
        self.out_channels = out_channels

        for i in range(1, n_blocks):
            layers.append(block(
                in_channels=self.out_channels, out_channels=out_channels
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer0(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

if __name__ == '__main__':
    tensor = torch.rand([1, 3, 224, 224])
    model = ResNet(n_channels=3, n_layers=18, block=SkipBlock, n_classes=1000)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    output = model(tensor)
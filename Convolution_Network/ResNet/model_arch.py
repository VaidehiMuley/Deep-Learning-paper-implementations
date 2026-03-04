import torch
import torch.nn as nn
import torchvision.transforms as transforms

## 50 layer ResNet implementation
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        '''
        Section 3.4 : We adopt batch normalization (BN) right after each convolution and before activation
        '''
        super().__init__()
        self.conv = nn.Conv2d(
                in_channels = in_channels,
                out_channels= out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
                )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        '''
        Function to stack the above defined layers
        '''
        return self.bn(self.conv(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels , first):
        super().__init__()
        stride = 1
        res_channels = out_channels//4 ## !check
        ## Either the input and output can have same dimensions or we will have to do projection
        self.projection = in_channels != out_channels
        if self.projection:
            ## Do projection
            self.p = ConvBlock(in_channels, out_channels, 1, 2, 0)
            stride = 2

        if first:
            self.p = ConvBlock(in_channels=in_channels,
                               out_channels=out_channels,
                               filter_size = 1,
                               stride = 1, ## Stride = 1 during the first layer of residual block - since we only want to change channel dimensions and not spatial dimensions
                               padding = 0
                               )

        ## Residual Bottleneck
        ## 1X1
        self.conv1 = ConvBlock(in_channels = in_channels,  ## We downsize the channels and keep spatial size same - Bottleneck design
                               out_channels = res_channels, ## shrink down 4 times
                               filter_size = 1,
                               stride = 1,
                               padding = 0
                               )

        # 3X3
        self.conv2 = ConvBlock(
            in_channels= res_channels,
            out_channels= res_channels,
            filter_size = 3,
            stride = stride, # We can make stride = 2 here since this is 3X3 and looks at neighboring pixels so meaningful downsample
            padding = 0
            )

        # 1X1
        self.conv3 = ConvBlock(
            in_channels = res_channels,
            out_channels = out_channels,  ## Expands back 4 times
            filter_size = 1,
            stride = 1,
            padding = 0
        )
        self.relu = nn.ReLU()


    def forward(self,x):
        f = self.relu(self.conv1(x))
        f = self.relu(self.conv2(f))
        f = self.conv3(f)

        if self.projection:
            x = self.p(x)
        return self.relu(torch.add(f,x))

class ResNet(nn.Module):
    def __init__(self, in_channels = 3, classes = 1000):
        super().__init__()


        self.conv1 = ConvBlock(in_channels,
                               out_channels = 64,
                               kernel_size=7,
                               stride = 2,
                               padding= 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride= 2, padding = 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) ## Used for averaging and also flattens before fc layer
        self.fc = nn.Linear(in_features= 2048, out_features = classes)
        self.relu = nn.ReLU()

        self.blocks = nn.ModuleList(ResidualBlock(64, 256, True)) # First block

        self.block_counts = [3,4,6,3] # Number of Residual blocks in each stage
        self.output_shapes = [256, 512, 1024, 2048] # Output shape of each residual block in each stage

        for stage in range(4): ## Loop over the number of stages
            ## We already defined first block of first stage, so skip it
            if stage > 0:
                self.blocks.append(ResidualBlock(in_channels= self.output_shapes[stage -1],
                                                 out_channels = self.output_shapes[stage]))
            for _ in range(self.output_shapes[stage]-1):
                self.blocks.append(ResidualBlock(in_channels=self.output_shapes[stage], out_channels=self.output_shapes[stage]))
        self.init_weight()

    def forward(self, x):
        ## Input size : 224 X 224
        x = self.relu(self.conv1(x))
        ## output size = ((W - k + 2P)/S) + 1 = ((224 - 7 + 6)/ 2)  + 1= 112 X 112

        ## Input size = 112 X 112
        x = self.maxpool(x)
        ## Output size = 56 X 56

        ## Input size = 56 X 56
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.relu(x)
        return x

        def init_weight(self):
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)


if __name__ == "__main__":
    config_name = 50 # 50-layer
    resnet50 = ResNet(50)
    image = torch.rand(1, 3, 224, 224)
    print(resnet50(image).shape)

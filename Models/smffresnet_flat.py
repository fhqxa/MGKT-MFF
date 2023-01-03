import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class MFF(nn.Module):
    '''
    多尺度特征融合 MFF
    '''

    def __init__(self, channels=640, r=4):
        super(MFF, self).__init__()
        inter_channels = int(channels // r)
        kernel_size1 = 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size1, stride=1, padding=0, bias=False)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            # avg =  avg_out = torch.mean(x, dim=1, keepdim=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        avg_out = torch.mean(xlg, dim=1, keepdim=True)
        max_out, _ = torch.max(xlg, dim=1, keepdim=True)
        xlg = torch.cat([avg_out, max_out], dim=1)
        xlg = self.conv1(xlg)
        wei = self.sigmoid(xlg)*xlg 
        xo = x * wei + residual * (1 - wei)
        return xo


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, fuse_type='MFF'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride

        if fuse_type == 'MFF':
            self.fuse_mode = MFF(channels=planes)
        else:
            self.fuse_mode = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        print('out_residual',out.shape,residual.shape)
        out = self.fuse_mode(out, residual)
        out = self.relu(out)
        out = self.maxpool(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes, block=BasicBlock, fuse_type='MFF'):  
        self.inplanes = 3
        super(ResNet, self).__init__()
        assert fuse_type in ['MFF']
        self.layer1 = self._make_layer(block, 64, stride=2, fuse_type=fuse_type)
        self.layer2 = self._make_layer(block, 160, stride=2, fuse_type=fuse_type)
        self.layer3 = self._make_layer(block, 320, stride=2, fuse_type=fuse_type)
        self.layer4 = self._make_layer(block, 640, stride=2, fuse_type=fuse_type)
        self.linear = nn.Linear(640, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, fuse_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, fuse_type=fuse_type))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = F.avg_pool2d(x, x.size()[3])

        x = x.view(x.size(0), -1)
        
        out = self.linear(x) 

        return out



def resnet32(num_classes):
    return ResNet(num_classes=num_classes, block=BasicBlock, fuse_type='MFF')
    
def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


# resnet = resnet32(100)
# a = torch.randn(1,3,32,32)
# resnet(a)
#
# test(resnet)

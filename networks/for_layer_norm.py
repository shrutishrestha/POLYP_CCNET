#replace in ccnet_769
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools


import sys, os

#from cc_attention import CrissCrossAttention
from .CC import CC_module as CrissCrossAttention
from utils.pyt_utils import load_model

from inplace_abn import InPlaceABN, InPlaceABNSync
from Synchronized.sync_batchnorm import SynchronizedBatchNorm2d as SyncBN
# BatchNorm2d = SyncBN#functools.partial(InPlaceABNSync, activation='identity')
LayerNorm = torch.nn.LayerNorm
def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, input_size, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = LayerNorm(input_size)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = LayerNorm(input_size)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = LayerNorm(input_size)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.input_size = input_size

    def forward(self, x):

        residual = x 
        out = self.conv1(x) #[1, 64, 193, 193]

        self.bn1 = LayerNorm(out.shape[2:]).cuda()
        out = self.bn1(out) #[1, 64, 193, 193]

        out = self.relu(out) #[1, 64, 193, 193]

        out = self.conv2(out) #[1, 64, 193, 193]

        self.bn2 = LayerNorm(out.shape[2:]).cuda()
        out = self.bn2(out) #[1, 64, 193, 193]

        out = self.relu(out) #[1, 64, 193, 193]

        out = self.conv3(out) #[1, 256, 193, 193]

        self.bn3 = LayerNorm(out.shape[2:]).cuda()
        out = self.bn3(out) #[1, 256, 193, 193]

        if self.downsample is not None:

            residual = self.downsample(x) #[1, 256, 193, 193]

        out = out + residual #[1, 256, 193, 193]

        out = self.relu_inplace(out) #[1, 256, 193, 193]

        return out


class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, input_size):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   LayerNorm(input_size),nn.ReLU(inplace=False))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   LayerNorm(input_size),nn.ReLU(inplace=False))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            LayerNorm(input_size),nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=1):
        output = self.conva(x) #[1, 64, 193, 193]

        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output) #[1, 64, 193, 193]
        output = self.bottleneck(torch.cat([x, output], 1)) #[1, 2, 193, 193]

        return output

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, criterion, recurrence):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = LayerNorm([385, 385])
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = LayerNorm([385, 385])
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = LayerNorm([385, 385])
        self.relu3 = nn.ReLU(inplace=False)


        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0],  input_size=(193, 193))
        self.layer2 = self._make_layer(block, 128, layers[1], input_size=(193, 193), stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], input_size=(97, 97), stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], input_size=(97, 97), stride=1, dilation=4, multi_grid=(1,1,1))
        #self.layer5 = PSPModule(2048, 512)
        self.head = RCCAModule(2048, 512, num_classes, [97, 97])
        self.head1 = RCCAModule(256, 64, num_classes,  [193,193])


        self.inv_128_64 = nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1)
        self.bn64_input = nn.LayerNorm([769, 769])
        self.conv_reduce_64_1 = nn.Conv2d(64,1,stride=1,kernel_size=1)

        self.inv_256_128_x1 = nn.ConvTranspose2d(256,128, kernel_size=3,stride=2, padding=1)
        self.bn128_x1 = nn.LayerNorm([385, 385])
        self.inv_128_64_x1 = nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1)
        self.bn64_x1= nn.LayerNorm([769, 769])
        self.conv_reduce_64_1_x1 = nn.Conv2d(64,1,stride=1,kernel_size=1)

        self.inv_2048_1024_x4 = nn.ConvTranspose2d(2048,1024,kernel_size=3,stride=2,padding=1)
        self.inv_1024_512_x4 = nn.ConvTranspose2d(1024,512,kernel_size=3,stride=2,padding=2)
        self.inv_512_256_x4 = nn.ConvTranspose2d(512,256, kernel_size=3,stride=2, padding=1)

        self.bn1024_x4 = nn.LayerNorm([194, 194])
        self.bn512_x4 = nn.LayerNorm([385, 385])
        self.bn256_x4 = nn.LayerNorm([769, 769])

        self.conv_reduce_256_1_x4 = nn.Conv2d(256,1,stride=1,kernel_size=1)

        self.conv_reduce_3_2 = nn.Conv2d(3,2,stride=1,kernel_size=1)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            LayerNorm([97, 97]),nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.criterion = criterion
        self.recurrence = recurrence

    def _make_layer(self, block, planes,  blocks, input_size, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride == 2:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                LayerNorm([97,97]))

        elif stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                LayerNorm(input_size))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, input_size = input_size, stride=stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, input_size = input_size, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x, labels=None): #[2, 3, 769, 769]
        x = self.relu1(self.bn1(self.conv1(x))) #[2, 64, 385, 385]
        x = self.relu2(self.bn2(self.conv2(x))) #[2, 64, 385, 385]

        x_input = self.relu3(self.bn3(self.conv3(x))) #[2, 128, 385, 385]

        x_maxpool = self.maxpool(x_input) #([1, 128, 193, 193])
        x1 = self.layer1(x_maxpool) #[1, 2, 193, 193]

        x_head1 = self.head1(x1, self.recurrence) #[1, 2, 193, 193]

        x2 = self.layer2(x1) #[2, 512, 97, 97])

        x3 = self.layer3(x2) #[2, 1024, 97, 97]

        x_dsn = self.dsn(x3) #[2, 2, 97, 97]

        x4 = self.layer4(x3) #[2, 2048, 97, 97]

        x_head = self.head(x4, self.recurrence) #[2, 2, 97, 97]

        #xinput
        xinput_inv = self.relu(self.bn64_input(self.inv_128_64(x_input, output_size=(769,769))))  #[1, 1, 769, 769]
        xinput_inv = self.conv_reduce_64_1(xinput_inv) #[1, 1, 769, 769]

        # #x1
        x1_inv = self.relu(self.bn128_x1(self.inv_256_128_x1(x1,output_size=(385,385)))) #x1 [1, 128, 385, 385]

        x1_inv = self.relu(self.bn64_x1(self.inv_128_64_x1(x1_inv, output_size=(769,769)))) #[1, 64, 769, 769]

        x1_inv = self.conv_reduce_64_1_x1(x1_inv)   #[1, 1, 769, 769]


        # #x4
        
        x4_inv = self.relu(self.bn1024_x4(self.inv_2048_1024_x4(x4,output_size=(194,194)))) #x4 [1, 1024, 194, 194]

        x4_inv = self.relu(self.bn512_x4(self.inv_1024_512_x4(x4_inv,output_size=(385,385)))) #[1, 512, 385, 385]

        x4_inv = self.relu(self.bn256_x4(self.inv_512_256_x4(x4_inv, output_size=(769,769)))) #[1, 256, 769, 769]

        x4_inv = self.conv_reduce_256_1_x4(x4_inv) #[1, 1, 769, 769]

        x_inv_cat = torch.cat([xinput_inv, x1_inv, x4_inv], 1) #[1, 3, 769, 769]

        x_upsampled_dsn =  self.conv_reduce_3_2(x_inv_cat) #1, 2, 769, 769]

        ccnet_head1 = F.interpolate(input=x_head1, size=(769, 769), mode='bilinear', align_corners=True) #[1, 2, 769, 769]

        ccnet_head = F.interpolate(input=x_head, size=(769, 769), mode='bilinear', align_corners=True) #[1, 2, 769, 769]

        ccnet_out = ccnet_head1 + ccnet_head #[1, 2, 769, 769]

        outs = [ccnet_out, x_upsampled_dsn]

        return outs

def Seg_Model(backbone, num_classes, continue_training, criterion, pretrained_model=None, recurrence=2, **kwargs):
    if continue_training:
        print("continue model state")
        model = ResNet(Bottleneck,[3, 4, 6, 3], num_classes, criterion, recurrence)

        if pretrained_model is not None:
            model = load_model(model, pretrained_model)
    else:
        print("starting model state")
        if backbone == "ResNet-50":
            layer = [3, 4, 6, 3]
            pretrained_file = "resnet50-imagenet.pth"

        elif backbone == "ResNet-101":
            layer = [3, 4, 23, 3]
            pretrained_file = "resnet101-imagenet.pth"

        print("backbone",backbone,"layer",layer)
        model = ResNet(Bottleneck,layer, num_classes, criterion, recurrence)

        # if pretrained_model is not None:
        #     file_path = os.path.join(pretrained_model, pretrained_file)
        #     model = load_model(model, file_path)



    return model
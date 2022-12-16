import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from layers import *
import numpy as np

class nn4(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(nn4, self).__init__()
        
        self.conv1 = conv_block(in_channels = in_channels, out_channels = 64, kernel_size = 7, stride = 2, padding = 3)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(64)
        
        self.inception_2 = Inception_block(64, (0, 64, 192, 0, 0, 0))
        
        self.norm2 = nn.BatchNorm2d(192)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception_3a = Inception_block(192, (64, 96, 128, 16, 32, 32), pool_type="max")
        self.inception_3b = Inception_block(256, (64, 96, 128, 32, 64, 64), pool_type="L2")
        self.inception_3c = Inception_block(320, (0, 128, 256, 32, 64, 0), stride_tup=(2, 2, 2), pool_type="max")
        
        self.inception_4a = Inception_block(640, (256, 96, 192, 32, 64, 128), pool_type="L2")
        self.inception_4b = Inception_block(640, (224, 112, 224, 32, 64, 128), pool_type="L2")
        self.inception_4c = Inception_block(640, (192, 128, 256, 32, 64, 128), pool_type="L2")
        self.inception_4d = Inception_block(640, (160, 144, 288, 32, 64, 128), pool_type="L2")
        self.inception_4e = Inception_block(640, (0, 160, 256, 64, 128, 0), stride_tup=(2, 2, 2), pool_type="max")
        
        self.inception_5a = Inception_block(1024, (384, 192, 384, 0, 0, 128), pool_type="L2")
        self.inception_5b = Inception_block(896, (384, 192, 384, 0, 0, 128), pool_type="max")
        
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(896, 128)
        
        # # in_channels, filter_num, stride_tup=(1, 1, 1), pad_tup=(1,2,1), pool_type=None, dim_red=0, out_pool=(3, 1)
        # #filter_num = (out_1x1, red_3x3, out_3x3, red_5x5, out_5x5)
        # #stride = (out_3x3, out_5x5, out_pool)
        # #padding = (out3x3, out5x5, out_pool)
        # #out_pool = (kernel_size_pool, stride_pool)
        # #pool_type = "max" or "L2"
        # self.inception2 = Inception_block(False, 64, 0, 64, 192, 0, 0, 0, pool='max')
        
    def forward(self, x):
        x=self.conv1(x)
        
        x=self.maxpool1(x)
        x=self.norm1(x)
        
        x=self.inception_2(x)
        
        x=self.norm2(x)
        x=self.maxpool2(x)
        
        x=self.inception_3a(x)
        x=self.inception_3b(x)
        x=self.inception_3c(x)
        
        x=self.inception_4a(x)
        x=self.inception_4b(x)
        x=self.inception_4c(x)
        x=self.inception_4d(x)
        x=self.inception_4e(x)
        
        x=self.inception_5a(x)
        x=self.inception_5b(x)
        
        x=self.avgpool1(x)
        
        x=self.fc1(torch.flatten(x))
        
        x=F.normalize(x, dim=0) #L2 normalization
        
        return x

if __name__== '__main__':
    # for i in range(0, 1000):
    x = torch.randn(1, 3, 96, 96)
    model = nn4()
    print(model(x).shape)
    print(model(x))
    # print(i)
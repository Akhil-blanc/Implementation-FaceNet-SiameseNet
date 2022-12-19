import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class nn1(nn.Module):
    def __init__(self):
        super(nn1, self).__init__()
        self.convlayers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(5),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.Conv2d(64, 192, kernel_size=3, stride=1),
            nn.ReLU(inplace=False),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 192, kernel_size=1, stride=1),
            nn.Conv2d(192, 384, kernel_size=3, stride=1,padding="same"),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2,padding=1),
            nn.Conv2d(384, 384, kernel_size=1, stride=1,padding="same"),
            nn.Conv2d(384, 256, kernel_size=3, stride=1,padding="same"),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=1, stride=1,padding="same"),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding="same"),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=1, stride=1,padding="same"),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding="same"),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2,padding=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(12544, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 128)
        )

    def forward(self, x):
        conv_output = self.convlayers(x)
        flattened_output = nn.Flatten(1)(conv_output)
        fc_output = self.fc(flattened_output)
        norm_output = nn.functional.normalize(fc_output)
        output = nn.Flatten(1)(norm_output)
        return output

class nn2(nn.Module):
    """nn2 Class is used to create an Inception network object with the architecture of NN2 (inception_224x224) given in "FaceNet". It takes as input a RGB image of size 224x224x3.
    
    """
    def __init__(self, in_channels=3, num_classes=1000):
        """nn2 Class constructor to initialize the obect.

        Args:
            in_channels (int, optional): depth of input image. (Must be RGB)
                                         Defaults to 3.
            num_classes (int, optional): number of individual classes(faces). Defaults to 1000.
        """
        
        super(nn2, self).__init__()
        
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
        
        self.inception_5a = Inception_block(1024, (384, 192, 384, 48, 128, 128), pool_type="L2")
        self.inception_5b = Inception_block(1024, (384, 192, 384, 48, 128, 128), pool_type="max")
        
        self.avgpool1 = nn.AvgPool2d(kernel_size=7, stride=1)
        
        self.fc1 = nn.Linear(1024, 128)
        
    def forward(self, x):
        """_summary_

        Args:
            x (tensor): an input image (RGB) of size 224x224x3

        Returns:
            array: an embedding of size 128
        """
        
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
        print(x.shape)
        
        x=self.fc1(torch.flatten(x, start_dim=1))
        
        x=F.normalize(x, dim=0) #L2 normalization
        
        return x
    
class nn3(nn.Module):
    """nn2 Class is used to create an Inception network object with the architecture of NN2 (inception_224x224) given in "FaceNet". It takes as input a RGB image of size 224x224x3.
    
    """
    def __init__(self, in_channels=3, num_classes=1000):
        """nn3 Class constructor to initialize the obect.

        Args:
            in_channels (int, optional): depth of input image. (Must be RGB)
                                         Defaults to 3.
            num_classes (int, optional): number of individual classes(faces). Defaults to 1000.
        """
        
        super(nn3, self).__init__()
        
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
        
        self.inception_5a = Inception_block(1024, (384, 192, 384, 48, 128, 128), pool_type="L2")
        self.inception_5b = Inception_block(1024, (384, 192, 384, 48, 128, 128), pool_type="max")
        
        self.avgpool1 = nn.AvgPool2d(kernel_size=5, stride=1)
        
        self.fc1 = nn.Linear(1024, 128)
        
    def forward(self, x):
        """_summary_

        Args:
            x (tensor): an input image (RGB) of size 160x160x3

        Returns:
            array: an embedding of size 128
        """
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
        
        x=self.fc1(torch.flatten(x, start_dim=1))
        
        x=F.normalize(x, dim=0) #L2 normalization
        
        return x

class nn4(nn.Module):
    """nn2 Class is used to create an Inception network object with the architecture of NN2 (inception_224x224) given in "FaceNet". It takes as input a RGB image of size 224x224x3.
    
    """
    def __init__(self, in_channels=3, num_classes=1000):
        """nn2 Class constructor to initialize the obect.

        Args:
            in_channels (int, optional): depth of input image. (Must be RGB)
                                         Defaults to 3.
            num_classes (int, optional): number of individual classes(faces). Defaults to 1000.
        """
        
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
        
    def forward(self, x):
        """_summary_

        Args:
            x (tensor): an input image (RGB) of size 94x94x3

        Returns:
            array: an embedding of size 128
        """
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
        
        x=self.fc1(torch.flatten(x, start_dim=1))
        
        x=F.normalize(x, dim=0) #L2 normalization
        
        return x

class nns1(nn.Module):
    """nn2 Class is used to create an Inception network object with the architecture of NN2 (inception_224x224) given in "FaceNet". It takes as input a RGB image of size 224x224x3.
    
    """
    def __init__(self, in_channels=3, num_classes=1000):
        """nn2 Class constructor to initialize the obect.

        Args:
            in_channels (int, optional): depth of input image. (Must be RGB)
                                         Defaults to 3.
            num_classes (int, optional): number of individual classes(faces). Defaults to 1000.
        """
        
        super(nns1, self).__init__()
        
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
        self.inception_4e = Inception_block(640, (0, 160, 256, 64, 128, 0), stride_tup=(2, 2, 2), pool_type="max")
        
        self.inception_5a = Inception_block(1024, (256, 96, 384, 0, 0, 96), pool_type="L2")
        self.inception_5b = Inception_block(736, (256, 96, 384, 0, 0, 96), pool_type="max")
        
        self.avgpool1 = nn.AvgPool2d(kernel_size=6, stride=1)
        
        self.fc1 = nn.Linear(736, 128)
       
    def forward(self, x):
        """_summary_

        Args:
            x (tensor): an input image (RGB) of size 165x165x3

        Returns:
            array: an embedding of size 128
        """
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
        x=self.inception_4e(x)
        
        x=self.inception_5a(x)
        x=self.inception_5b(x)
        
        x=self.avgpool1(x)
        
        x=self.fc1(torch.flatten(x, start_dim=1))
        
        x=F.normalize(x, dim=0) #L2 normalization
        return x

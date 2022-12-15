import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class Inception_block(nn.Module):
    def __init__(self, in_channels, filter_num, stride_tup=(1, 1, 1), pad_tup=(1,2,1), pool_type=None, pool_kernel=3):
        super(Inception_block, self).__init__()
        
        self.pad_tup = pad_tup
        self.pool_type = pool_type
        #filter_num = (out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_dim_red)
        #stride = (out_3x3, out_5x5, out_pool)
        #padding = (out3x3, out5x5, out_pool)
        #out_pool = (kernel_size_pool, stride_pool)
        #pool_type = "max" or "L2"
        
        self.branches = []
        
        if filter_num[0]:
            self.branch1 = conv_block(in_channels, filter_num[0], kernel_size=1)
            self.branches.append(1)
            
        if filter_num[2]:
            if filter_num[1]:
                self.branch2 = nn.Sequential(
                    conv_block(in_channels, filter_num[1], kernel_size=1),
                    conv_block(filter_num[1], filter_num[2], kernel_size=3, stride=stride_tup[0], padding=pad_tup[0])
                )
                self.branches.append(2)
            else:
                raise ValueError("Invalid filter size for reduction layer.")

        if filter_num[4]:
            if filter_num[3]:
                self.branch3 = nn.Sequential(
                    conv_block(in_channels, filter_num[3], kernel_size=1),
                    conv_block(filter_num[3], filter_num[4], kernel_size=5, stride=stride_tup[1], padding=pad_tup[1])
                )
                self.branches.append(3)
            else:
                raise ValueError("Invalid filter size for reduction layer.")
            
        #creating layer 4
        if pool_type == "max":    
            self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=pool_kernel, stride=stride_tup[2], padding=pad_tup[2]),
            )
            self.branches.append(4)
        elif pool_type == "L2":
            self.branch4 = nn.Sequential(
                nn.LPPool2d(norm_type=2, kernel_size=pool_kernel, stride=stride_tup[2])
            )
            self.branches.append(4)
        elif pool_type == None:
            None
        else:
            raise ValueError("Invalid Pooling Layer Type:", pool_type)
        
        if filter_num[5]:
            self.branch4.append(conv_block(in_channels, filter_num[5], kernel_size=1))
        
    
    def forward(self, x):
        concat_list = []
        # print(self.branches)
        for i in range(1, 5):
            if i in self.branches:
                # print('inside')
                if i == 1:
                    concat_list.append(self.branch1(x))
                elif i == 2:
                    concat_list.append(self.branch2(x))
                elif i == 3:
                    concat_list.append(self.branch3(x))
                else:
                    if self.pool_type == "L2":
                        # print("l2")
                        # print(F.pad(x, (self.pad_tup[2], self.pad_tup[2], self.pad_tup[2], self.pad_tup[2]), "constant", 0).shape)
                        concat_list.append(self.branch4(F.pad(x, (self.pad_tup[2], self.pad_tup[2], self.pad_tup[2], self.pad_tup[2]), "constant", 0)))
                    else:
                        # print("max")
                        concat_list.append(self.branch4(x))
                        
        # for i in self.branches:
        #     if i == self.branch4:
        #         concat_list.append(i(F.pad(x, self.pad_tup[2], "constant", 0)))
        #     else:
        #         concat_list.append(i(x))
                
        return torch.cat(concat_list, 1)

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception_block(nn.Module):
    """Inception_block Class is used to create an Inception module object similar to the Inception module described in the Fig 2(b) of "Going Deeper with Convolutions" paper.
        
    The two major differences are:
        1. There is an option to choose which layers to apply from the four layers generally used.
        2. Pooling can be chosen to be "Max Pool" or "L2 Pool", and also dimensionality reduction can be applied if needed.
    """
    
    def __init__(self, in_channels, filter_num, stride_tup=(1, 1, 1), pad_tup=(1,2,1), pool_type=None, pool_kernel=3):
        """Inception_block Class constructor to initialize the obect.

        Args:
            in_channels (int): depth of input map
            filter_num (tuple): contains depth of output map from layers in the order (out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool)
                                give integer '0' in tuple to avoid the respective layer
            stride_tup (tuple, optional): contains stride value for layers in the order (out_3x3, out_5x5, out_pool). 
                                          Defaults to (1, 1, 1).
            pad_tup (tuple, optional): contains padding applied in layers in the order (out_3x3, out_5x5, out_pool).
                                       Defaults to (1,2,1).
            pool_type (string, optional): "max" or "L2" based on pooling preference.
                                          Defaults to None.
            pool_kernel (int, optional): kernel size of the pooling layer.
                                         Defaults to 3.
        """
        super(Inception_block, self).__init__()
        
        #defining variables to be used throughout the class
        self.pad_tup = pad_tup
        self.pool_type = pool_type   
        self.branches = []      #list to store the branches that has to be used in the inception module object
        
        #creating 1x1 convolution layer if specified
        if filter_num[0]:
            self.branch1 = conv_block(in_channels, filter_num[0], kernel_size=1)
            self.branches.append(1)
            
        #creating 1x1 reduction and 3x3 convolution layers if specified
        if filter_num[2]:
            if filter_num[1]:
                self.branch2 = nn.Sequential(
                    conv_block(in_channels, filter_num[1], kernel_size=1),                                                  #reduction layer
                    conv_block(filter_num[1], filter_num[2], kernel_size=3, stride=stride_tup[0], padding=pad_tup[0])       #convolution layer
                )
                self.branches.append(2)
            else:
                raise ValueError("Invalid filter size for reduction layer.")

        #creating 1x1 reduction and 5x5 convolution layers if specified
        if filter_num[4]:
            if filter_num[3]:
                self.branch3 = nn.Sequential(
                    conv_block(in_channels, filter_num[3], kernel_size=1),                                                  #reduction layer
                    conv_block(filter_num[3], filter_num[4], kernel_size=5, stride=stride_tup[1], padding=pad_tup[1])       #convolution layer
                )
                self.branches.append(3)
            else:
                raise ValueError("Invalid filter size for reduction layer.")
            
        #creating pooling layer if specified
        if pool_type == "max":    
            self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=pool_kernel, stride=stride_tup[2], padding=pad_tup[2])            #max pool layer
            )
            self.branches.append(4)
        elif pool_type == "L2":
            self.branch4 = nn.Sequential(
                nn.LPPool2d(norm_type=2, kernel_size=pool_kernel, stride=stride_tup[2])                 #L2 pool layer                     
            )
            self.branches.append(4)
        elif pool_type == None:
            None
        else:
            raise ValueError("Invalid Pooling Layer Type:", pool_type)
        
        #adding dimensionality reduction (for filter number) if given
        if filter_num[5]:
            self.branch4.append(conv_block(in_channels, filter_num[5], kernel_size=1))                  #convolution layer
        
    
    def forward(self, x):
        """Concatenates the processed layers from the Inception module along their depth.

        Args:
            x (tensor): 3D HWD input map

        Returns:
            Tensor: Inception layer processed 3D maps.
        """
        
        concat_list = []        #list of branches to be given as input to torch.cat()

        for i in range(1, 5):
            if i in self.branches:
                if i == 1:
                    concat_list.append(self.branch1(x))
                elif i == 2:
                    concat_list.append(self.branch2(x))
                elif i == 3:
                    concat_list.append(self.branch3(x))
                else:
                    if self.pool_type == "L2":
                        #zero padding input tensor to apply L2 poolinf without change in height and width of input map
                        x = F.pad(x, (self.pad_tup[2], self.pad_tup[2], self.pad_tup[2], self.pad_tup[2]), "constant", 0)       
                        
                        concat_list.append(self.branch4(x))
                    else:
                        concat_list.append(self.branch4(x))
        
        return torch.cat(concat_list, 1)

class conv_block(nn.Module):
    """conv_block Class is used to create a 2D Convolution layer with batch normalization object.
    
    After applying 2D Convolution, batch normalization and ReLU activation has been applied.

    """
    def __init__(self, in_channels, out_channels, **kwargs):
        """conv_block Class constructor to initialize the obect.

        Args:
            in_channels (int): depth of input map
            out_channels (int): depth of output map
        """
        
        super(conv_block, self).__init__()
        
        self.relu = nn.ReLU()                                               #creating ReLU activation function object
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)          #creating 2D Convolution layer object
        self.batchnorm = nn.BatchNorm2d(out_channels)                       #creating Batch Normalization function object
        
    def forward(self, x):
        """Return the 2D convoluted, batch normalized and activation applied output map from the input map.

        Args:
            x (tensor): 3D HWD input map

        Returns:
            Tensor: 2D convoluted, batch normalized and activation applied 3D maps.
        """
        
        return self.relu(self.batchnorm(self.conv(x)))
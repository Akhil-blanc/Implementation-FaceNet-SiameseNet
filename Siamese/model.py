import torch
import torch.nn as nn

#create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        #Setting up the Sequential of the CNN layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 10, stride = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride = 2),

            nn.Conv2d(64, 128, kernel_size = 7, stride = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride = 2),

            nn.Conv2d(128, 128, kernel_size = 4, stride = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride = 2),

            nn.Conv2d(128, 256, kernel_size = 4, stride = 1),
            nn.ReLU(inplace=True),

            nn.Flatten(1, -1)
        )
        #Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.Sigmoid()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        #This function will be declared for both the images
        #It's output is used to determine the similarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        #Here in this function we pass in both the images and obtain the corresponding vectors that are returned.
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        L1_dist = torch.sub(output1, output2)

        L1_dist = torch.abs(L1_dist)
       
        output = self.fc2(L1_dist)

        return output
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.5, std=0.01)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.2)
                nn.init.normal_(m.bias, mean=0.5, std=0.01)

siamese_model = SiameseNetwork()
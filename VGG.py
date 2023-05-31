# Import some important packages
import torch 
import torch.nn as nn # All neural networks module, nn.Linear, nn.Conv2D, BatchNorm, LossFunction 
import torch.optim as optim # For all OPtimization algorithm, Sgd, Adam, etc..
import torch.nn.functional as F # All functions that don not have any parameters
from torch.utils.data import DataLoader # Give easier dataset managment and creates mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way 
import torchvision.transforms as transforms # Transformations we can perform on our dataset

# VGG16 architeture config
VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# then flatten this

class VGGNet(nn.Module): 
    def __init__(self, in_channels = 3, num_classes = 1000):
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layer(VGG16)

        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
            )
    

    def forward(self): 
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layer(self, architecture): 
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
                
        return nn.Sequential(*layers)



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGGNet(in_channels=3,num_classes=1000).to(device)
    print(model)
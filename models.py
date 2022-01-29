import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator1d(nn.Module):
    def __init__(self, inSize, midSize, outSize):
        super(Generator1d, self).__init__()
        self.inSize = inSize
        self.outSize = outSize
        self.midSize = midSize

        self.model = nn.Sequential(
            nn.Linear(self.inSize, self.midSize, bias = True),
            nn.Softplus(),
            nn.Linear(self.midSize, self.midSize, bias = True),
            nn.Softplus,
            nn.Linear(self.midSize, self.outSize, bias = True),
        )

    def forward(self,x):
        return self.model(x)


class Discriminator1d(nn.Module):
    
    def __init__(self, inSize, midSize, outSize):
        super(Discriminator1d,self).__init__()
        self.inSize = inSize
        self.outSize = outSize
        self.midSize = midSize

        self.model = nn.Sequential(
            nn.Linear(self.inSize, self.midSize, bias = True),
            nn.Tanh(),
            nn.Linear(self.midSize, self.outSize, bias = True)
        )
    
    def forward(self,x):
        return self.model(x)

#pentru Degenset: featureSize = 256, inSize = 100, channelSize = 3 
class Generator2d(nn.Module):
    def __init__(self, inSize, featureSize, channelSize):
        super(Generator2d, self).__init__()
        self.inSize = inSize
        self.featureSize = featureSize
        self.channelSize = channelSize

        self.model = nn.Sequential(
            nn.ConvTranspose2d(inSize,featureSize * 8, 4, 2, 0, bias=False),
            nn.BatchNorm2d(featureSize * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(featureSize * 8,featureSize * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(featureSize * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(featureSize * 4,featureSize * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(featureSize * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(featureSize * 2,featureSize, 4, 2, 1, bias=False),
            nn.BatchNorm2d(featureSize),
            nn.ReLU(True),

            nn.ConvTranspose2d(featureSize,channelSize, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)


#pentru Degenset channelSize = 3, featureSize = 256, outSize = 1
class Discriminator2d(nn.Module):
    
    def __init__(self, channelSize, featureSize, outSize):
        super(Discriminator2d,self).__init__()
        self.channelSize = channelSize
        self.featureSize = featureSize
        self.outSize = outSize

        self.model = nn.Sequential(
            nn.Conv2d(channelSize,featureSize, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(featureSize, featureSize * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(featureSize * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(featureSize * 2, featureSize * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(featureSize * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(featureSize * 4, featureSize * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(featureSize * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(featureSize * 8, outSize, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)

class Gan :
    def __init__(self):
        self.disc = Discriminator2d(channelSize= 3,featureSize=64,outSize= 1).to(device)
        self.gen = Generator2d(channelSize= 3, featureSize= 64,inSize= 100).to(device)

        
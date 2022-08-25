from queue import Queue
import torch.nn as nn
import torch
import torch.optim as optim
import sys

class pcgGenerator(nn.Module):
    def __init__(self,inSize, midSize, outSize):
        super(pcgGenerator,self).__init__()
        self.inSize = inSize
        self.midSize = midSize
        self.outSize = outSize

        self.model = nn.Sequential(
            nn.Linear(self.inSize, self.inSize//2),
            nn.Softplus(),
            nn.Linear(self.inSize//2, self.inSize//4),
            nn.Softplus(),
            nn.Linear(self.inSize//4, self.outSize)
        )
    def forward(self,x):
        return self.model(x) + torch.rand(self.outSize).to(device) -0.5 

    def forwardSample(self):
        return "," + str(self.model(torch.rand(self.inSize)*6 -3).tolist())[1:-1]
        
        

class pcgDiscriminator(nn.Module):
    def __init__(self, inSize, midSize, outSize):
        super(pcgDiscriminator,self).__init__()
        self.inSize = inSize
        self.midSize = midSize
        self.outSize = outSize

        self.model = nn.Sequential(
            nn.Linear(self.inSize, self.midSize//2),
            nn.Tanh(),
            nn.Linear(self.midSize//2, self.outSize),
        )
    
    def forward(self,x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gradientPenalty(disc, real, fake):
    L= real.shape
    epsilon = torch.rand((1)).repeat(L).to(device)
    interpolation = real * epsilon + fake * (1 - epsilon)

    mixedScores = disc.forward(interpolation)
    gradient= torch.autograd.grad(
        inputs= interpolation,
        outputs= mixedScores,
        grad_outputs= torch.ones_like(mixedScores),
        create_graph= True,
        retain_graph= True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim = 1)
    return torch.mean((gradient_norm -1) ** 2)

def train(
    data,
    disc,
    gen,
    optimD,
    optimG,
    discReps = 5,
    genReps = 1,
    clampLower=-0.01,
    clampHigher=0.01,
    nrEpochs = 5,
    ):
    disc.to(device)
    gen.to(device)
    for _ in range(nrEpochs):
                reals = torch.FloatTensor(data).to(device)
                for _ in range (discReps):
                    fake = gen.forward(torch.randn(gen.inSize).to(device))
                    disc_fake = disc.forward(fake)
                    disc_real = disc.forward(reals)
                    #gp = gradientPenalty(disc,reals,fake)
                    discLoss = torch.mean(disc_fake)-torch.mean(disc_real) #+ 0.1 * gp
                    
                    disc.zero_grad()
                    discLoss.backward()
                    optimD.step()

                    for p in disc.parameters():
                        p.data.clamp_(clampLower,clampHigher)
                
                for _ in range(genReps):
                    fake = gen.forward(torch.randn(gen.inSize).to(device))
                    disc_fake= disc.forward(fake)

                    genLoss = -torch.mean(disc_fake)
                    #print(genLoss,file=sys.stderr)
                    gen.zero_grad()
                    genLoss.backward()
                    optimG.step()
    disc.cpu()
    gen.cpu()

class Gan():
    #aici tre stabilit cum intra si cum iese
    def __init__(self,key,fSize):
        self.key = key
        self.fSize = fSize
        self.disc = pcgDiscriminator(fSize,fSize,1)
        self.gen = pcgGenerator(100,fSize,fSize)
        self.optimD = optim.RMSprop(self.disc.parameters(), lr = 0.0005)
        self.optimG = optim.RMSprop(self.gen.parameters(), lr = 0.0005)
        self.queue = Queue()



    

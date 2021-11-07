from torch import random
import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.nn.modules.activation import Softplus
from PIL import Image
import loader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gradientPenalty(disc, real, fake):
    batchSize, C, H , W= real.shape
    epsilon = torch.rand((batchSize,1,1,1)).repeat(1,C,H,W).to(device)
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
    batchSize,
    data,
    disc,
    gen,
    discReps,
    genReps,
    learnRate = 0.001,
    clampLower=-0.01,
    clampHigher=0.01,
    nrEpochs = 5,
    ):
    optimD = optim.RMSprop(disc.parameters(), lr = learnRate)
    optimG = optim.RMSprop(gen.parameters(), lr = learnRate)
    for _ in range(nrEpochs):
        for  _,reals in enumerate(data):
            reals = reals.to(device)
            for _ in range (discReps):
                fake = gen.forward(torch.rand(reals.shape[0],100,1,1).to(device))
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
                fake = gen.forward(torch.rand(reals.shape[0],100,1,1).to(device))
                disc_fake= disc.forward(fake)

                genLoss = -torch.mean(disc_fake)
                gen.zero_grad()
                genLoss.backward()
                optimG.step()
          #  discLosses.append(discLoss.cpu().detach().numpy())
        
    #modify Generator and Discriminator. Dodon o dat link-uri.

            
                
        


import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.nn.modules.activation import Softplus
from PIL import Image
import loader
import numpy as np



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
    optimC = optim.RMSprop(disc.parameters(), lr = learnRate)
    optimG = optim.RMSprop(gen.parameters(), lr = learnRate)

    for _ in range(nrEpochs):
        for  _,reals in enumerate(data):
            for _ in range (discReps):
                fake = gen.forward(torch.rand(25,100,1,1))
                disc_fake = disc.forward(fake)
                disc_real = disc.forward(reals)
                discLoss = torch.mean(disc_fake)-torch.mean(disc_real)

                disc.zero_grad()
                discLoss.backward()
                optimC.step()
                for p in disc.parameters():
                    p.data.clamp_(clampLower,clampHigher)
            
            for _ in range(genReps):
                fake = gen.forward(torch.rand(25,100,1,1))
                disc_fake= disc.forward(fake)

                genLoss = -torch.mean(disc_fake)
                gen.zero_grad()
                genLoss.backward()
                optimG.step()
    #modify Generator and Discriminator. Dodon o dat link-uri.

            
                
        


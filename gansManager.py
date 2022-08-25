import copy
import threading
import PCG_Models
import warnings
import sys
import torch
import os
import shutil
warnings.filterwarnings("ignore")
#variables
gans = {}
generators ={} 

#classes
class lockedGens():
    def __init__(self,gen):
        self.lock = threading.Lock()
        self.gen = gen
    def setGen(self,gen):
        self.gen = gen

#functions
def trainCore(gan):
    while True:
        data = gan.queue.get()
        PCG_Models.train(
            data = data,
            disc = gan.disc,
            gen = gan.gen,
            optimD= gan.optimD,
            optimG= gan.optimG
            )
        gen = generators[gan.key]
        gen.lock.acquire()
        generators[gan.key].setGen(copy.deepcopy(gan.gen)) 
        gen.lock.release()

def requestTrain(gan, data):
    gan.queue.put(data)

def makeGan(key,outSize):
    gans[key] = PCG_Models.Gan(key,outSize)
    generators[key] = lockedGens(copy.deepcopy(gans[key].gen))
    threading.Thread(target= trainCore, args = [gans[key],]).start()
    print('g'+ str(key))
    sys.stdout.flush()

def genForward(key):
    gen = generators[key]
    gen.lock.acquire()
    toReturn = gen.gen.forwardSample()
    gen.lock.release()
    print(str(key) + toReturn)
    sys.stdout.flush()

def save(categName):
    dir = "./categories/" + categName
    if not os.path.exists(dir):
        os.mkdir(dir)
    for item in gans.items(): 
        torch.save({
            str(item[0])+'g': item[1].gen.state_dict(),
            str(item[0])+'d': item[1].disc.state_dict(),
            str(item[0])+'oD': item[1].optimD.state_dict(),
            str(item[0])+'oG': item[1].optimG.state_dict()
        },dir + '/model'+ str(item[0]) + '_' + str(item[1].fSize) + '.pt')

def load(categName):
    dir = "./categories/" + categName
    for fName in os.listdir(dir):
        fullFName = os.path.join(dir,fName)
        loaded = torch.load(fullFName);
        key,fSize = fName[5:-3].split('_')
        key = int(key)
        fSize = int(fSize)
        gans[key] = PCG_Models.Gan(key,fSize)
        gans[key].disc.load_state_dict(loaded[str(key) + 'd'])
        gans[key].gen.load_state_dict(loaded[str(key) + 'g'])
        gans[key].optimD.load_state_dict(loaded[str(key)+'oD'])
        gans[key].optimG.load_state_dict(loaded[str(key)+'oG'])
        generators[key] = lockedGens(copy.deepcopy(gans[key].gen))
        threading.Thread(target= trainCore, args = [gans[key],]).start()
    sys.stdout.flush()
    print("s")

def clearAll():
    gans.clear()
    generators.clear()

def getAllCategs():
    toReturn = 'c '
    dir = "./categories/"
    for fName in os.listdir(dir):
        toReturn = toReturn+ fName + ','
    print(toReturn[:-1])
def deleteCateg(categName):
    clearAll()
    shutil.rmtree("./categories/" + categName);
    
#main
while(True):
    inp = input().split(sep=" ")
    if inp[0] == "m":
        makeGan(int(inp[1]),int(inp[2]))
    elif inp[0] == "t":
        requestTrain(gans[int(inp[1])],[float(x) for x in inp[2].split(sep=",")],)
    elif inp[0] == "g":
        threading.Thread(target = genForward, args = [int(inp[1])] ).start()
    elif inp[0] == "s":
        save(inp[1])
        sys.stdout.flush()
        print('s')
    elif inp[0] == "l":
        load(inp[1])
    elif inp[0] == "d":
        clearAll()
        print('s')
    elif inp[0] == "c":
        getAllCategs()
    elif inp[0] == "x":
        deleteCateg(inp[1])
    else: 
        break

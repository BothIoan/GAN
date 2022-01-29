import copy
import threading
import PCG_Models
import warnings
import sys
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
            gen = gan.gen
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


#main
while(True):
    inp = input().split(sep=" ")
    if inp[0] == "m":
        makeGan(int(inp[1]),int(inp[2]))
    elif inp[0] == "t":
        requestTrain(gans[int(inp[1])],[float(x) for x in inp[2].split(sep=",")],)
    elif inp[0] == "g":
        threading.Thread(target = genForward, args = [int(inp[1])] ).start()
    else: 
        break

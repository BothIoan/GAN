import torch.utils.data as data
from PIL import Image
from IPython.display import display
import os
import os.path
import numpy as np

class Dataset(data.Dataset):
    def __init__(self,img_list):
        super(Dataset,self).__init__()
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,idx):
        img= self.img_list[idx]
        return img


def imgAppend(dir):
    images = []
    for img in os.listdir(dir):
        imgPath = os.path.join(dir,img)
        size = 128,128
        image = np.array(Image.open(imgPath).resize((64,64),Image.LANCZOS))
        #H W C
        image = image.astype('float32')
        image = image.transpose(2,0,1)
        #C H W
        image /=255
        images.append(image)
    return images


def imgDisplay(data):
    img = Image.fromarray(data,'RGB')
    img.show()


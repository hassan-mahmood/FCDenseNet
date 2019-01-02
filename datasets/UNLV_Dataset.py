
from torch.utils.data import Dataset
import csv
import os
from PIL import Image
import csv
import torch
import numpy as np

class UNLV_Dataset(Dataset):
    def __init__(self,imagespath,labelspath,transforms):
        self.imagespath=imagespath
        self.labelspath=labelspath
        self.img_names=os.listdir(self.imagespath)
        self.transforms=transforms


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        imgname=self.img_names[idx]
        #img=self.transforms(Image.open(os.path.join(self.imagespath,imgname)).convert('RGB'))
        img = self.transforms(Image.open(os.path.join(self.imagespath, imgname)).convert('L'))

        #coordconv
        _, height, width = img.size()
        a = np.tile(np.array(np.arange(width) / width), (height, 1))
        b = np.tile(np.array([np.arange(height) / height]).transpose(), (1, width))
        coord = torch.from_numpy(np.stack((a, b))).type(torch.FloatTensor)
        stacked = torch.cat((img, coord), 0)

        label=self.transforms(Image.open(os.path.join(self.labelspath,imgname)).convert('L'))

        return stacked,label,imgname


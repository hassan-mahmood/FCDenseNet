
from torch.utils.data import Dataset
import csv
import os
from PIL import Image
import csv
import torch

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
        img=self.transforms(Image.open(os.path.join(self.imagespath,imgname)).convert('RGB'))
        label=self.transforms(Image.open(os.path.join(self.labelspath,imgname)).convert('L'))

        return img,label,imgname


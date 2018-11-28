import torch
import torchvision
import cv2
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as fn
import torch
import numpy as np
from math import ceil
import csv
import shutil



dirpath='../data/images/'
outpath='../data/images2/'
files=os.listdir(dirpath)

for f in files:
    im=cv2.imread(os.path.join(dirpath,f))
    im=im[:,:,0]
    #im=cv2.resize(im,(512,512))
    orig_im=im
    kernel = np.ones((10,10),np.uint8)
    im = cv2.erode(im,kernel,iterations = 1)
    im=im.T

    kernel = np.ones((10,10),np.uint8)
    im = cv2.erode(im,kernel,iterations = 1)

    im=im.T
    #cv2.imwrite(os.path.join('data/labels/',f),im)
    cv2.imwrite(os.path.join(outpath,f), im)
    print('Done ',f)


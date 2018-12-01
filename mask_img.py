import cv2
from scipy.misc import imresize

import numpy as np
import os
import cv2

resultsdir='../masks/'
outdir='../outputs/'
files=os.listdir(resultsdir)

for file in files:
    img = cv2.imread(os.path.join('../images/',file))
    mask = cv2.imread(os.path.join(resultsdir,file),0)
    mask[mask != 255] = 0
    width,height=mask.shape
    img = cv2.resize(img,(height,width))
    img[:, :, 2][mask == 0] = 255
    #img[mask==[0,0,0]]=255
    #cv2.imshow('Figure',img)
    cv2.imwrite(os.path.join(outdir,file),img)
    print('Done ',file)


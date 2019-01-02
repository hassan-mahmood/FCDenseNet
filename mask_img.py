import cv2
from scipy.misc import imresize

import numpy as np
import os
import cv2

resultsdir='../masks/'
outdir='../outputs/'
indir='../images/'

if(not os.path.exists(outdir)):
    os.mkdir(outdir)

if(not os.path.exists(resultsdir)):
    os.mkdir(resultsdir)

if(not os.path.exists(indir)):
    print('Input path does not exist')
else:
    files=os.listdir(resultsdir)

    for file in files:
        img = cv2.imread(os.path.join(indir,file))
        mask = cv2.imread(os.path.join(resultsdir,file),0)
        mask[mask != 255] = 0
        width,height=mask.shape
        img = cv2.resize(img,(height,width))
        img[:, :, 2][mask == 0] = 255
        #img[mask==[0,0,0]]=255
        #cv2.imshow('Figure',img)
        cv2.imwrite(os.path.join(outdir,file),img)
        print('Done ',file)


import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm

parser=argparse.ArgumentParser()
parser.add_argument('--imagespath',default='data/Image/')
parser.add_argument('--outpath',default='data/Image2')
args=parser.parse_args()


dirpath=args.imagespath
outpath=args.outpath
if(not os.path.exists(outpath)):
    os.mkdir(outpath)

files=os.listdir(dirpath)

for f in tqdm(files):
    im=cv2.imread(os.path.join(dirpath,f))
    im=im[:,:,0]
    #im=cv2.resize(im,(512,512))
    orig_im=im
    kernel = np.ones((10,10),np.uint8)
    im = cv2.erode(im,kernel,iterations = 1)
    # im=im.T
    #
    # kernel = np.ones((10,10),np.uint8)
    # im = cv2.erode(im,kernel,iterations = 1)
    #
    # im=im.T
    # #cv2.imwrite(os.path.join('data/labels/',f),im)
    cv2.imwrite(os.path.join(outpath,f), im)



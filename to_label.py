

import csv
import cv2
from PIL import Image
import os
import numpy as np
import shutil
import json

dirpath='data/'

files=os.listdir(os.path.join(dirpath,'images/'))
trainlen=int(len(files)*0.8)
splitdata=dict()
splitdata['train']=files[:trainlen]
splitdata['val']=files[trainlen:]

def makedir(path):
    if(not os.path.exists(path)):
        os.mkdir(path)

makedir(os.path.join(dirpath,'train'))
makedir(os.path.join(dirpath,'val'))
makedir(os.path.join(dirpath,'train','images'))
makedir(os.path.join(dirpath,'train','labels'))
makedir(os.path.join(dirpath,'val','images'))
makedir(os.path.join(dirpath,'val','labels'))


for t in ['val','train']:
    for file in splitdata[t]:
        if(os.path.exists(os.path.join(dirpath,'unlv_xml_gt/',file.split('.')[0]+'.json'))):
            shutil.move(os.path.join(dirpath,'images',file),os.path.join(dirpath,t,'images',file))

    subimgpath=os.path.join(dirpath,t)
    for imgname in os.listdir(os.path.join(subimgpath,'images')):
        img = cv2.imread(os.path.join(subimgpath,'images', imgname))
        h, w, _ = img.shape
        outimg = np.zeros((h, w))
        with open(os.path.join(dirpath,'unlv_xml_gt/', imgname.split('.')[0] + '.json'), encoding='utf-8') as data_file:

            for data in json.loads(data_file.read()):
                ymin = int(data["top"])
                xmin = int(data["left"])
                ymax = int(data["bottom"])
                xmax = int(data["right"])
                outimg[ymin:ymax, xmin:xmax] = 255
            cv2.imwrite(os.path.join(os.path.join(subimgpath,'labels/'), imgname), outimg)
            print('Done ', imgname)




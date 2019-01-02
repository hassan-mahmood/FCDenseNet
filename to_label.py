

import csv
import cv2
from PIL import Image
import os
import numpy as np
import shutil
import json
import argparse
from tqdm import tqdm

parser=argparse.ArgumentParser()


parser.add_argument('--xmlspath',default='unlv_xml_gt/',help='Directory with all xml files')
parser.add_argument('--imagespath',default='images/',help='Directory containing all the images')
parser.add_argument('--outdir',default='data/',help='Output directory to store train and val data')




args=parser.parse_args()



files=os.listdir(args.imagespath)
trainlen=int(len(files)*0.8)
splitdata=dict()
splitdata['train']=files[:trainlen]
splitdata['val']=files[trainlen:]

def makedir(path):
    if(not os.path.exists(path)):
        os.mkdir(path)

makedir(args.outdir)
makedir(os.path.join(args.outdir,'train'))
makedir(os.path.join(args.outdir,'val'))
makedir(os.path.join(args.outdir,'train','images'))
makedir(os.path.join(args.outdir,'train','labels'))
makedir(os.path.join(args.outdir,'val','images'))
makedir(os.path.join(args.outdir,'val','labels'))


for t in ['val','train']:
    for file in splitdata[t]:
        if(os.path.exists(os.path.join(args.xmlspath,file.split('.')[0]+'.json'))):
            shutil.move(os.path.join(args.imagespath,file),os.path.join(args.outdir,t,'images',file))

    subimgpath=os.path.join(args.outdir,t)
    for imgname in tqdm(os.listdir(os.path.join(subimgpath,'images'))):
        img = cv2.imread(os.path.join(subimgpath,'images', imgname))
        h, w, _ = img.shape
        outimg = np.zeros((h, w))
        with open(os.path.join(args.xmlspath, imgname.split('.')[0] + '.json'), encoding='utf-8') as data_file:

            for data in json.loads(data_file.read()):
                ymin = int(data["top"])
                xmin = int(data["left"])
                ymax = int(data["bottom"])
                xmax = int(data["right"])
                outimg[ymin:ymax, xmin:xmax] = 255
            cv2.imwrite(os.path.join(os.path.join(subimgpath,'labels/'), imgname), outimg)
            #print('Done ', imgname)




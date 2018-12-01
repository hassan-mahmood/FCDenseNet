from sklearn.metrics import precision_recall_fscore_support, classification_report,accuracy_score

import cv2
import os
import numpy as np
from tqdm import tqdm

targetmasks='../other/targetmasks/'
outputmasks='../other/masks/'

outputfiles=os.listdir(outputmasks)
count=len(outputfiles)
f1_sum=0
p_sum=0
r_sum=0
acc=0

for i,imgname in tqdm(enumerate(outputfiles)):
    output=cv2.imread(os.path.join(outputmasks,imgname),0)
    output[output==255]=0
    output[output!=0]=255
    width,height=output.shape


    target=cv2.imread(os.path.join(targetmasks,imgname),0)
    target=cv2.resize(target,(height,width))

    output=output.flatten()
    target=target.flatten()
    # cv2.imshow('Figure',output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # cv2.imshow('Figure', target)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    results=precision_recall_fscore_support(target, output, average='micro')

    f1_sum +=float(results[2])
    p_sum+=float(results[0])
    r_sum+=float(results[1])
    acc+=accuracy_score(target, output)
    #print(classification_report(target,output))


print('F1: ',f1_sum/count)

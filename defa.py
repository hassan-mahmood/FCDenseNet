# # from PIL import Image
# # import os
# # import numpy as np
# # import cv2
# # import torch
# # import torchvision
# #
# # mytransforms=torchvision.transforms.Compose([torchvision.transforms.Resize((512)),torchvision.transforms.ToTensor()])
# #
# #
# # im=mytransforms(Image.open('img.png').convert('L'))
# # _,height,width=im.size()
# #
# #
# # a=np.tile(np.array(np.arange(width)/width), (height, 1))
# #
# # b=np.tile(np.array([np.arange(height)/height]).transpose(), (1, width))
# # coord=torch.from_numpy(np.stack((a,b))).type(torch.FloatTensor)
# # stacked=torch.cat((im,coord),0)
# # print(stacked.size())
# #
# import cv2
# import numpy as np
# from scipy import signal
# from matplotlib import pyplot as plt
# from skimage.util.shape import view_as_windows
#
# def strided4D_v2(arr,arr2,s):
#     return view_as_windows(arr, arr2.shape, step=(s,s))
#
# def stride_conv_strided(arr,arr2,s):
#     arr4D = strided4D_v2(arr,arr2,s=s)
#     return np.tensordot(arr4D, arr2, axes=((2,3),(0,1)))
#
#
# img=cv2.imread('img.png',0)
# #img=cv2.resize(img,(512,512))
#
# #filter=np.ones((20,20))
# #out=stride_conv_strided(img,filter,50)
# #out=signal.convolve(img,filter,'same')
#
#
# #cv2.imshow('image',laplacian)
#
# #color=np.dstack((img,img,img))
#
# #img=cv2.Laplacian(img,cv2.CV_64F)
# #lap2=cv2.Laplacian(lap1,cv2.CV_64F)
#
# fy=[[1,1,1],
#     [0,0,0],
#     [1,1,1]]
#
# fx=[[1,0,1],
#     [1,0,1],
#     [1,0,1]]
#
# length=5
# mid=int(length/2)
#
# filter=np.zeros((length,length),np.uint8)
#
# filter[mid,:]=1
# filter[:,mid]=1
#
#
# #derivx=signal.convolve(img,fx,'same')
# #derivy=signal.convolve(img,fy,'same')
# derivy=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# derivx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# derivx=cv2.erode(derivx,filter)
# derivy=cv2.erode(derivy,filter)
#
# #img=cv2.bitwise_not(img)
# dist=cv2.distanceTransform(img,cv2.DIST_L2,5)
# cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
# final=dist
# final=np.array(final>0.1,np.uint8)
# # final=np.array(final*255,np.uint8)
# # final=np.array((final>100)*128)
# # final=np.random.random((100,100))
# # final=np.array(final*255,np.uint8)
# # final=np.array(final>100,np.uint8)
# final=final*255
# print(final)
# #final=np.array((dist>0.1)*255)
#
# #_, final = cv2.threshold(dist, 0.15, 1.0, cv2.THRESH_BINARY)
#
# #final2=cv2.distanceTransform(img,cv2.DIST_L1,5)
# #final=np.multiply(derivx,derivy)
# #final=np.dot(derivx,derivy)
#
# cv2.namedWindow('image2',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image2', 600,600)
# #cv2.namedWindow('image2',cv2.WINDOW_NORMAL)
# #cv2.resizeWindow('image2', 600,600)
#
# cv2.imshow('image2',final)
# #cv2.imshow('image2',derivy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
#
# #cv2.imwrite('houghlines3.jpg',zeros)

import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)

im=cv2.imread('1.png',0)
#im=cv2.bitwise_not(im)
#orig_im=im
kernel = np.ones((10,20),np.uint8)
im = cv2.erode(im,kernel,iterations = 1)
#im=cv2.distanceTransform(im,cv2.DIST_L2,3)
cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.waitKey(0)
#cv2.destroyAllWindows()
# im=cv2.imread('mask.png',0)
# ret, markers = cv2.connectedComponents(im)
# zeros=np.zeros(markers.shape)
# print(zeros.shape)
#
# for num in range(1,ret):
#     loc=np.argwhere(markers==num)
#     print('Start: ',loc[0])
#     print('End: ',loc[-1])
#
# zeros[markers==1]=128
# zeros[markers==2]=255
#
#
# cv2.imshow('image',np.array(zeros,np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
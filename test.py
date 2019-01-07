import time

import torchvision
from torch.utils.data import DataLoader
from datasets.UNLV_Dataset import UNLV_Dataset as Dataset
import argparse
from models.tiramisu import *
import torch
import torch.nn as nn
import utils.training as train_utils
import os


parser=argparse.ArgumentParser()

parser.add_argument('--testdir',default='unlvdata/test/',help='Directory containing test images and labels')
parser.add_argument('--batchsize',default=2,help='Batchsize for training')
parser.add_argument('--weightfile',help='Weight file to use')
parser.add_argument('--classes',default=2)
args=parser.parse_args()


mytransforms=torchvision.transforms.Compose([torchvision.transforms.Resize((512)),torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomVerticalFlip(),torchvision.transforms.ToTensor()])

test_dataset=Dataset(os.path.join(args.testdir,'Image/'),os.path.join(args.testdir,'LabeledImage/'),mytransforms)
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=True,num_workers=1)


model=FCDenseNet103(args.classes)

model = nn.DataParallel(model).cuda()

criterion = nn.NLLLoss2d()

train_utils.load_weights(model,'../weights-28-0.120-0.230.pth')


train_utils.view_sample_predictions(model,test_loader)




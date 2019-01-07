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

parser.add_argument('--traindir',default='unlvdata/train/',help='Directory containg training images and labels')
parser.add_argument('--valdir',default='unlvdata/val/',help='Directory containing validation images and labels')
#parser.add_argument('--imagesdir',default='data/images/',help='Directory containing all images')
#parser.add_argument('--labelsdir',default='Directory with all labels')
parser.add_argument('--epochs',default=100,help='Number of epochs to train the model')
parser.add_argument('--lr',default=0.0001,help='Learning rate to update the parameters')
parser.add_argument('--decay',default=0.995)
parser.add_argument('--batchsize',default=2,help='Batchsize for training')
parser.add_argument('--classes',default=2)

args=parser.parse_args()

if(not os.path.exists('weights/')):
    os.mkdir('weights/')



DECAY_EVERY_N_EPOCHS = 1

mytransforms=torchvision.transforms.Compose([torchvision.transforms.Resize((512)),torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomVerticalFlip(),torchvision.transforms.ToTensor()])

train_dataset=Dataset(os.path.join(args.traindir,'Image/'),os.path.join(args.traindir,'LabeledImage/'),mytransforms)
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=1)

val_dataset=Dataset(os.path.join(args.valdir,'Image/'),os.path.join(args.valdir,'LabeledImage/'),mytransforms)
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=True,num_workers=1)


model=FCDenseNet103(args.classes)

#model=model.cuda()
model = nn.DataParallel(model).cuda()

optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-5)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-4)
criterion = nn.NLLLoss2d()

train_utils.load_weights(model,'weights/weights-0-0.178-0.000.pth')

for epoch in range(args.epochs):

    print('\n _______________________________________________')
    trn_loss, trn_err = train_utils.train(model, train_loader, optimizer, criterion, epoch)
    print('Epoch {:d}\nTrain - Loss: {:.4f}'.format(epoch, trn_loss))

    train_utils.save_weights(model, epoch, float(trn_loss), 0)

    ### Adjust Lr ###
    train_utils.adjust_learning_rate(args.lr, args.decay, optimizer, epoch, DECAY_EVERY_N_EPOCHS)

    ### Validate ###
    train_utils.view_sample_predictions(model,val_loader)

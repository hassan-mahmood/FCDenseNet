
import torchvision
from torch.utils.data import DataLoader
from datasets.UNLV_Dataset import UNLV_Dataset as Dataset
import argparse
from models.tiramisu import *
import torch
import torch.nn as nn
import utils.training as train_utils
import time


parser=argparse.ArgumentParser()

parser.add_argument('--traincsv',default='data/train.csv',help='CSV with training images')
parser.add_argument('--valcsv',default='data/val.csv',help='CSV with val images')
parser.add_argument('--imagesdir',default='data/images/',help='Directory containing all images')
parser.add_argument('--labelsdir',default='Directory with all labels')
parser.add_argument('--epochs',default=100,help='Number of epochs to train the model')
parser.add_argument('--lr',default=0.0001,help='Learning rate to update the parameters')
parser.add_argument('--decay',default=0.995)
parser.add_argument('--batchsize',default=1,help='Batchsize for training')
parser.add_argument('--classes',default=2)

args=parser.parse_args()


DECAY_EVERY_N_EPOCHS = 1
mytransforms=torchvision.transforms.Compose([torchvision.transforms.Resize((512)),torchvision.transforms.ToTensor()])

train_dataset=Dataset('data/train/images/','data/train/labels/',mytransforms)
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=1)

val_dataset=Dataset('data/val/images','data/val/labels/',mytransforms)
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=True,num_workers=1)


model=FCDenseNet57(args.classes)
model=model.cuda()
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-4)
criterion = nn.NLLLoss2d()



for epoch in range(args.epochs):
    since = time.time()

    # trn_loss, trn_err = train_utils.train(model, train_loader, optimizer, criterion, epoch)
    # print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(epoch, trn_loss, 1 - trn_err))
    #
    # time_elapsed = time.time() - since
    # print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    train_utils.load_weights(model,'weights/weights-14-0.120-0.230.pth')
    ### Test ###
    train_utils.view_sample_predictions(model,val_loader,85)

    # val_loss, val_err = train_utils.test(model, val_loader, criterion, epoch)
    # print('Val - Loss: {:.4f} | Acc: {:.4f}'.format(val_loss, 1 - val_err))
    # time_elapsed = time.time() - since
    # print('Total Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

    ### Checkpoint ###
    #train_utils.save_weights(model, epoch, val_loss, val_err)
    #train_utils.save_weights(model, epoch, 0.12,0.23)

    ### Adjust Lr ###
    #train_utils.adjust_learning_rate(args.lr, args.decay, optimizer,epoch, DECAY_EVERY_N_EPOCHS)

import os
import sys
import math
import string
import random
import shutil

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from . import imgs as img_utils
from . import tools
import numpy as np

RESULTS_PATH = 'results/'
WEIGHTS_PATH = 'weights/'


def save_weights(model, epoch, loss, err):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w
    incorrect = preds.ne(targets).cpu().sum()
    err = incorrect/n_pixels
    return round(err.item(),5)

def cross_entropy2d(input, target,cuda):

    log_softmax=nn.LogSoftmax()
    nll_loss=nn.NLLLoss()
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)

    log_p = log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    if(cuda):
        target=target.type(torch.cuda.LongTensor)
    else:
        target = target.type(torch.LongTensor)

    loss = nll_loss(log_p, target)
    if(cuda):
        loss=loss.type(torch.cuda.FloatTensor)
    else:
        loss = loss.type(torch.FloatTensor)
    # if size_average:
    #     loss = loss.item()/torch.sum(mask).item()
    return loss

def train(model, trn_loader, optimizer, criterion, epoch):
    model.train()
    trn_loss = 0
    trn_error = 0
    for idx, data in tqdm(enumerate(trn_loader)):

        inputs = Variable(data[0])
        inputs=inputs.type('torch.cuda.FloatTensor')
        targets = Variable(data[1])
        targets=targets.squeeze(1)
        targets=targets.type('torch.cuda.LongTensor')
        # inputs = Variable(data[0].cuda())
        # targets = Variable(data[1].cuda())
        optimizer.zero_grad()
        output = model(inputs)
        #loss = criterion(output, targets)
        loss=cross_entropy2d(output,targets,True)
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()
        pred = get_predictions(output)
        trn_error += error(pred, targets.data.cpu())



    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    return trn_loss, trn_error

def test(model, test_loader, criterion, epoch=1):
    model.eval()
    test_loss = 0
    test_error = 0
    with torch.no_grad():
        for idx, data in tqdm(enumerate(test_loader)):
            data, target,imgname=data
            # data = Variable(data.cuda(), volatile=True)
            # target = Variable(target.cuda())
            data = Variable(data)
            data=data.type('torch.cuda.FloatTensor')
            target = Variable(target)
            target=target.squeeze(1)
            target=target.type('torch.cuda.LongTensor')
            output = model(data)
            #test_loss += criterion(output, target).data[0]
            test_loss += cross_entropy2d(output, target, True)
            pred = get_predictions(output)
            test_error += error(pred, target.data.cpu())

    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    return test_loss, test_error

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()

def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target,imgname in input_loader:
        # data = Variable(input.cuda(), volatile=True)
        # label = Variable(target.cuda())
        data = Variable(input, volatile=True)
        label = Variable(target)
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input,target,pred])
    return predictions

def view_sample_predictions(model, loader, n):

    label_trues, label_preds = [], []

    for idx,batch in tqdm(enumerate(loader)):
        input, target,imgname=batch
        imgname=imgname[0]
        input = Variable(input)
        input=input.type('torch.cuda.FloatTensor')
        target = target.type('torch.LongTensor')

        output = model(input)
        #print('Pred shape:', output.shape)
        output = output.data.max(1)[1].squeeze_(1).squeeze_(0)
        #print('Target shape:', target.shape)

        #print('Pred shape:', output.shape)
        output=output.type('torch.FloatTensor')

        #uncomment below line to save the results
        tools.labelTopng(output, os.path.join('results/',str(imgname)))
        output=output.type('torch.LongTensor')
        #print('Output shape:',output.shape)
        #for calculating accuracy
        #output = output.data.max(1)[1].squeeze_(1).squeeze_(0)
        label_trues.append(target.numpy())
        label_preds.append(output.numpy())
    metrics = tools.accuracy_score(label_trues, label_preds)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
                Accuracy: {0}
                Accuracy Class: {1}
                Mean IU: {2}'''.format(*metrics))


# inputs, targets = next(iter(loader))
    #
    # data = Variable(inputs)
    # data=data.type('torch.cuda.FloatTensor')
    #
    # # data = Variable(inputs.cuda(), volatile=True)
    # # label = Variable(targets.cuda())
    #
    # output = model(data)
    # pred = get_predictions(output)
    # batch_size = inputs.size(0)
    # for i in range(min(n, batch_size)):
    #     #img_utils.view_image(inputs[i])
    #     label = Variable(targets[i])
    #     label=label.squeeze(2)
    #     label = label.type('torch.LongTensor')
    #     print('Target shape:',label.shape)
    #     print('Pred shape:',pred[i].shape)
    #     tools.labelTopng(label,str(i)+'target.jpg')
    #     tools.labelTopng(pred[i], str(i) + 'pred.jpg')
    #     #img_utils.view_annotated(str(i)+'target.jpg',label,False)
    #     #img_utils.view_annotated(str(i)+'pred.jpg',pred[i],False)
    #

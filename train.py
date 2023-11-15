# -*- coding: utf-8 -*-
import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import time, datetime
import pdb, traceback

import cv2
# import imagehash
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b4')

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

from resnet import Resnet18
from PVTnet import pvt_v2_b1
from ViTnet import ViT
#from Cdiffion import train
from DDPM import CDF
#from efficientnet_pytorch import resnet18_2
from thop import profile
from ptflops import get_model_complexity_info
# input dataset
train_jpg = np.array(glob.glob('./Train/*/*.png'))


class QRDataset(Dataset):
    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        start_time = time.time()
        img = Image.open(self.train_jpg[index]).convert('RGB')


        if self.transform is not None:
            img = self.transform(img)
        label = 0
        if 'NC' in self.train_jpg[index]:
            label = 0

        elif 'AD' in self.train_jpg[index]:
            label = 1

        elif 'MCI' in self.train_jpg[index]:
            label = 2

        return img, torch.from_numpy(np.array(label))
        #return img, torch.from_numpy(np.array(int('NC' in self.train_jpg[index])))

    def __len__(self):
        return len(self.train_jpg)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""

    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# -----------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, common_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out
# class VisitNet(nn.Module):
#     def __init__(self):
#         super(VisitNet, self).__init__()
#     #     model1 = resnet18_1()
#     #     model2 = resnet18_2()
#     #     self.resnet1 = model1
#     #     self.resnet2 = model2
#     #     self.fc = nn.Sequential(
#     #         nn.Dropout(0.5),
#     #         MLP(512, 3)
#     #     )
#     # def forward(self, img):
#     #     out1 = self.resnet1(img)
#     #     out2 = self.resnet2(img)
#     #     x = 0.02 * ((torch.abs(out1 - out2)) ** 2) + torch.abs(out1/2+out2/2)
#     #     # x2 = 0.02 * ((torch.abs(out2 - out1)) ** 2) + torch.abs(out2)
#     #     # x=x1+x2
#     #     x = self.fc(x)
#     #     return x
#     #     model1 = models.ResNet(False)
#     #     self.resnet1 = model1
#         model1 = ResNet()
#         # model2 = resnet18_2()
#         self.resnet1 = model1
#         self.fc = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.Dropout(0.5),
#             nn.ReLU(True),
#             nn.Linear(512, 3),
#         )
#
#     def forward(self, img):
#         out1 = self.resnet1(img)
#         return out1

#-------------
class VisitNet(nn.Module):
    def __init__(self):
        super(VisitNet, self).__init__()
        model1 = Resnet18(3)
        self.resnet1 = model1
    def forward(self, img):
        out = self.resnet1(img)
        return out
#---------------------------------

def validate(val_loader, model,criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5)

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            #1
            ##############################################
            out = model(input)
            cbam = CBAM(channel=150)
            Foutput = cbam(out)
            loss = criterion(Foutput, target.long())
            # measure accuracy and record loss
            acc1, acc5 = accuracy(Foutput, target, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        return top1


def predict(test_loader, model, tta=10):
    # switch to evaluate mode
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(test_loader):
                input = input.cuda()
                target = target.cuda()

                # compute output
                #output = model(input, path)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta

###########################################multi-attention mechanism
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            # nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            # nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x=x.cuda()
        avgout = self.shared_MLP(self.avg_pool(x))
        #print(avgout.shape)
        maxout = self.shared_MLP(self.avg_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        # x=x.cuda()
        # x = x.reshape([-1, 150, 32, 32])
        out = self.channel_attention(x) * x
       # print('outchannels:{}'.format(out.shape))
        #out = self.spatial_attention(out) * out
        return out
#output149 = output149.cuda()
######################################################

def train(train_loader, model,criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1)

    # switch to train mode
    model.train()
    end = time.time()
    for ii, (input, target) in enumerate(train_loader):
        #print(train_loader.shape)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        #target = target
##############################################
        out = model(input)
#########################################################################
        loss = criterion(out, target.long())
        acc1, acc5 = accuracy(out, target, topk=(1, 2))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #print(ii)
        if ii % 25 == 0:
            #print("gg")
            progress.pr2int(ii)


skf = KFold(n_splits=10, random_state=233, shuffle=True)
for flod_idx, (train_idx, val_idx) in enumerate(skf.split(train_jpg, train_jpg)):
    train_loader = torch.utils.data.DataLoader(
        QRDataset(train_jpg[train_idx],
                  transforms.Compose([
                      # transforms.RandomGrayscale(),
                      transforms.Resize((96, 96)),
                      transforms.RandomAffine(10),
                      # transforms.ColorJitter(hue=.05, saturation=.05),
                      # transforms.RandomCrop((450, 450)),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
                  ), batch_size=150, shuffle=True, num_workers=0, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        QRDataset(train_jpg[val_idx],
                  transforms.Compose([
                      transforms.Resize((96, 96)),
                      # transforms.Resize((124, 124)),
                      # transforms.RandomCrop((450, 450)),
                      # transforms.RandomCrop((88, 88)),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
                  ), batch_size=150, shuffle=False, num_workers=0, pin_memory=True
    )

    class VisitNet(nn.Module):
        def __init__(self):
            super(VisitNet, self).__init__()
            model1 = Resnet18(3)
            model2 = ViT(
                image_size=96,
                patch_size=32,
                num_classes=3,
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
            )
            self.resnet1 = model1
            self.dsvit = model2
        def forward(self, img):
            out2 = self.resnet1(img)
            out1 = self.dsvit(img, out2)
            Foutput = torch.mean(torch.stack([out1, out2]), 0)
            cbam = CBAM(channel=150)
            out = cbam(Foutput)
            return out
    #---------------------------------
    model = VisitNet().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    #1
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)

    best_acc = 0.0
    for epoch in range(100):
        scheduler.step()
        print('Epoch: ', epoch)

        train(train_loader, model, criterion, optimizer, epoch)
        val_acc = validate(val_loader, model, criterion)

        if val_acc.avg.item() > best_acc:
            best_acc = val_acc.avg.item()
            torch.save(model.state_dict(), './model/resnet18_fold{0}.pt'.format(flod_idx))

    break